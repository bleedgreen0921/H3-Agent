"""Small explicit real-dependency checks. No mock model and no automatic installation."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from ai_assistant.config import load_config, SkillsConfig


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


async def models(config):
    from ai_assistant.knowledge.models import embedding, reranker

    require(config.knowledge is not None, "需要 knowledge 配置。")
    async with embedding(config.knowledge) as model:
        vectors = await model.aembed_documents(
            ["中国的首都是北京。", "苹果是一种水果。"]
        )
        require(len(vectors) == 2, "Embedding 批量响应数量错误。")
        await model.aembed_query("中国的首都是哪里？")
    scores = await asyncio.to_thread(
        reranker(config.knowledge).score,
        [
            ("中国的首都是哪里？", "苹果是一种水果。"),
            ("中国的首都是哪里？", "中国的首都是北京。"),
        ],
    )
    require(
        len(scores) == 2 and scores[1] > scores[0], "真实重排未将相关文档排在前面。"
    )
    return {
        "dimensions": len(vectors[0]),
        "scores": scores,
        "scope": "真实模型服务；不包含数据库或 Agent",
    }


async def rag(config, args):
    from ai_assistant.knowledge.config import resolve_knowledge_paths
    from ai_assistant.knowledge.ingestion import submit, run_once
    from ai_assistant.knowledge.retrieval import create_knowledge_retriever
    from ai_assistant.storage.rag import RagStore
    from ai_assistant.workspace.files import ReadOnlyWorkspace

    require(config.knowledge is not None, "需要 knowledge 配置。")
    base = config.config_path.parent
    c = resolve_knowledge_paths(config.knowledge, base)
    with TemporaryDirectory(prefix="assistant-rag-check-") as directory:
        root = Path(directory)
        token = "CODE-" + uuid4().hex
        (root / "sample.csv").write_text(
            "项目,校验码\n星河迁移," + token + "\n", encoding="utf-8"
        )
        c = c.model_copy(
            update={
                "knowledge_base_id": "rag-check-" + uuid4().hex,
                "materials_dir": root / "materials",
                "diagnostics_dir": root / "diagnostics",
            }
        )
        store = RagStore(c)
        job = None
        try:
            job = await submit(
                store, ReadOnlyWorkspace(root), "sample.csv", tags=["smoke"]
            )
            require(
                await submit(
                    store, ReadOnlyWorkspace(root), "sample.csv", tags=["smoke"]
                )
                == job,
                "重复提交未复用任务。",
            )
            await run_once(store)
            state = await store.status(job)
            require(
                state["state"] == "succeeded", "入库未完成：" + str(state.get("error"))
            )
            docs = await create_knowledge_retriever(
                c, filters={"tags": ["smoke"]}
            ).ainvoke("星河迁移的校验码是什么？")
            require(
                docs and any(token in d.page_content for d in docs),
                "真实检索或重排未找到随机校验码。",
            )
            require(
                all(d.id and d.metadata.get("document_version_id") for d in docs),
                "缺少可追踪来源。",
            )
            result = {
                "scope": "真实解析、后台入库、混合检索及重排；不包含生成模型回答",
                "sources": len(docs),
            }
            if args.agent:
                from ai_assistant.runtime import assistant_session, final_text

                selected = config.model_copy(
                    update={
                        "knowledge": c,
                        "skills": None,
                        "mcp": None,
                        "workspace": config.workspace.model_copy(
                            update={"root": str(root)}
                        ),
                        "tools": config.tools.model_copy(
                            update={
                                "terminal": None,
                                "web": None,
                                "files": config.tools.files.model_copy(
                                    update={"enabled": False}
                                ),
                            }
                        ),
                    }
                )
                async with assistant_session(selected, args.model) as session:
                    turn = await session.run_turn(
                        "检索知识库，报告星河迁移的校验码，并引用片段 ID。"
                    )
                    require(
                        turn.sources
                        and any(token in d.page_content for d in turn.sources),
                        "Agent 未返回本次资料的真实检索来源。",
                    )
                    require(
                        token in final_text(turn.messages),
                        "Agent 回答未包含随机校验码。",
                    )
                    result["scope"] = (
                        "真实解析、后台入库、检索重排及生成模型 Agent 回答"
                    )
            return result
        finally:
            if job:
                state = await store.status(job)
                if state:
                    await store.delete(state["document_id"])
            await store.close()


async def agent(config, capability, args):
    from ai_assistant.runtime import assistant_session
    from ai_assistant.config import TerminalConfig
    from ai_assistant.skills.catalog import SkillCatalog
    from types import SimpleNamespace
    from langchain_core.messages import ToolMessage

    if capability == "feishu":
        require(
            args.prompt and args.expect and args.required_tool,
            "飞书验收需显式提供 --prompt、--expect、--required-tool，写入请求中必须指定测试目标。",
        )
        selected, prompt = config, args.prompt
        snapshot = None
    else:
        snapshot = None
        temp = TemporaryDirectory(prefix="assistant-agent-check-")
        root = Path(temp.name)
        token = "CHECK-" + uuid4().hex
        (root / "source.txt").write_text(token)
        selected = config.model_copy(
            update={
                "workspace": config.workspace.model_copy(update={"root": str(root)}),
                "knowledge": None,
                "skills": None,
                "mcp": None,
                "tools": config.tools.model_copy(
                    update={
                        "terminal": None,
                        "web": None,
                        "files": config.tools.files.model_copy(
                            update={"enabled": False}
                        ),
                    }
                ),
            }
        )
        prompt = "调用 read_text 读取 source.txt，原样回答文件中的校验码。"
        args.expect, args.required_tool = token, "read_text"
        if capability == "skills":
            packages = root / "packages"
            package = packages / "check-note"
            package.mkdir(parents=True)
            (package / "SKILL.md").write_text(
                "---\nname: check-note\ndescription: 校验本地资源\n---\n必须读取 rules.txt 并报告其中校验码。"
            )
            (package / "rules.txt").write_text(token)
            snapshot = SkillCatalog([packages])
            selected = selected.model_copy(
                update={"skills": SkillsConfig(roots=[str(packages)])}
            )
            prompt = "按已激活 check-note 的规则操作，不运行命令。"
            args.required_tool = "read_skill_resource"
        elif capability == "web":
            require(config.tools.web is not None, "需要 tools.web 配置。")
            selected = selected.model_copy(
                update={
                    "tools": selected.tools.model_copy(update={"web": config.tools.web})
                }
            )
            prompt = "调用 web_fetch 阅读 https://example.com，报告页面标题和来源链接。"
            args.expect, args.required_tool = "Example Domain", "web_fetch"
        elif capability == "terminal":
            selected = selected.model_copy(
                update={
                    "tools": selected.tools.model_copy(
                        update={"terminal": TerminalConfig()}
                    )
                }
            )
            prompt = (
                "只调用一次 run_command 执行 "
                + json.dumps("printf " + token)
                + "，原样回答输出。"
            )
            args.required_tool = "run_command"
    try:
        async with assistant_session(
            selected,
            args.model,
            resources=SimpleNamespace(skills=snapshot) if snapshot else None,
        ) as session:
            turn = await session.run_turn(
                prompt, skill_ids=["check-note"] if snapshot else None
            )
            results = [m for m in turn.messages if isinstance(m, ToolMessage)]
            matched = [
                m
                for m in results
                if m.name == args.required_tool
                or (
                    getattr(m, "additional_kwargs", {}).get("name")
                    == args.required_tool
                )
            ]
            # MCP 原生名称映射为会话工具 ID，允许用精确原生名称选择。
            if not matched and selected.mcp:
                directory = session.context.mcp
                ids = {
                    e.id
                    for e in directory.entries.values()
                    if e.original_name == args.required_tool
                }
                matched = [m for m in results if m.name in ids]
            require(
                matched and all(m.status == "success" for m in matched),
                "未观察到指定工具成功执行。",
            )
            require(args.expect in turn.messages[-1].text, "回答未包含预期核对值。")
            return {
                "scope": "真实生成模型与工具流程",
                "session_id": session.session_id,
                "tool": args.required_tool,
            }
    finally:
        if snapshot:
            snapshot.close()
        if capability != "feishu":
            temp.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capability",
        choices=("models", "rag", "chat", "skills", "terminal", "web", "feishu"),
    )
    parser.add_argument("--config", default="assistant.toml")
    parser.add_argument("--model")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="rag 同时验证真实生成模型回答，可能产生费用",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--expect")
    parser.add_argument("--required-tool")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    async def run():
        config = load_config(args.config)
        if args.capability == "models":
            return await models(config)
        if args.capability == "rag":
            return await rag(config, args)
        return await agent(config, args.capability, args)

    try:
        result = {"status": "passed", **asyncio.run(run())}
    except KeyboardInterrupt:
        print("已取消；外部效果不会自动回滚。", file=sys.stderr)
        return 130
    except Exception as error:
        # 不输出 SDK 原始异常；其中可能包含凭据。
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": "检查配置、依赖、服务和业务目标；本次未自动重跑。",
        }
        if isinstance(error, RuntimeError) and type(error) is RuntimeError:
            result["detail"] = str(error)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                args.output.chmod(0o600)
                stream.write(text + "\n")
        except OSError:
            print("结果文件无法创建或已存在，未覆盖。", file=sys.stderr)
            return 2
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
