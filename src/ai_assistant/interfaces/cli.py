"""Configuration validation and explicit live model checks."""

import argparse
import asyncio
import math
import sys

from ai_assistant.config import ConfigError, load_config
from ai_assistant.storage.sessions import SessionError


def _positive_timeout(value: str) -> float:
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Timeout must be a finite positive number.")


def _print_result(result) -> None:
    status = {"passed": "通过", "failed": "失败", "skipped": "未检查"}[result.status]
    usage = "未提供" if result.usage is None else ", ".join(f"{k}={v}" for k, v in result.usage.items())
    descriptions = {
        "thinking_mismatch": "服务返回了与禁用设置不符的推理内容",
        "incomplete_or_refused": "输出截断或请求被拒绝", "empty_output": "输出为空",
        "invalid_tool_call": "工具名称、数量或参数无效", "unexpected_tool_call": "返回了意外的工具调用",
        "invalid_output_or_parameters": "输出或请求参数不符合要求", "unsupported_interface": "当前集成未实现该接口",
        "authentication": "认证失败", "balance": "余额不足", "permission": "权限不足",
        "not_found": "模型或接口不存在", "rate_limit": "请求被限流", "http_error": "服务返回 HTTP 错误",
        "timeout": "检查超时", "connection": "连接失败", "request_failed": "请求失败",
    }
    detail = descriptions.get(result.error, "")
    if result.http_status is not None:
        detail += f" (HTTP {result.http_status})"
    print(f"{result.capability:<12} {status}  {result.elapsed_seconds:.2f}s  用量：{usage}  {detail}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    chat_parser = commands.add_parser("chat", help="Interactive assistant")
    chat_parser.add_argument("--config", default="assistant.toml", metavar="PATH")
    chat_parser.add_argument("--model", metavar="NAME")
    chat_parser.add_argument("--session", metavar="SESSION_ID")
    config_parser = commands.add_parser("config", help="Application configuration")
    config_commands = config_parser.add_subparsers(dest="action", required=True)
    check = config_commands.add_parser("check", help="Validate configuration without network access")
    check.add_argument("--config", default="assistant.toml", metavar="PATH")
    check.add_argument("--model", metavar="NAME")
    model_parser = commands.add_parser("model", help="Live model checks")
    model_commands = model_parser.add_subparsers(dest="action", required=True)
    probe = model_commands.add_parser("check", help="Send real requests using profile SDK retries")
    probe.add_argument("--config", default="assistant.toml", metavar="PATH")
    probe.add_argument("--model", metavar="NAME")
    probe.add_argument("--capability", choices=("all", "chat", "stream", "tools", "json_mode", "json_schema"), default="all")
    probe.add_argument("--timeout", type=_positive_timeout, default=60.0)
    from ai_assistant.knowledge.cli import add_commands
    add_commands(commands)
    from ai_assistant.interfaces.sessions import add_commands as add_sessions
    add_sessions(commands)
    from ai_assistant.skills.cli import add_commands as add_skills
    add_skills(commands)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "skill":
            from ai_assistant.skills.cli import execute
            from ai_assistant.skills.catalog import SkillError
            try:
                return asyncio.run(execute(config, args))
            except (SkillError, OSError, ImportError) as exc:
                print(f"Skill 操作失败：{exc}", file=sys.stderr)
                return 2
        if args.command == "rag":
            from ai_assistant.knowledge.cli import execute
            from ai_assistant.knowledge import KnowledgeError
            try:
                return asyncio.run(execute(config, args))
            except (KnowledgeError, ImportError, OSError) as exc:
                print(f"RAG 操作失败：{exc if isinstance(exc, KnowledgeError) else type(exc).__name__}", file=sys.stderr)
                return 2
            except Exception:
                print("RAG 操作失败：数据库或集成错误；请检查依赖、连接和显式迁移状态。", file=sys.stderr)
                return 2
        if args.command == "session":
            from ai_assistant.interfaces.sessions import execute
            try:
                return asyncio.run(execute(config, args))
            except SessionError as exc:
                print(f"会话操作失败：{exc}", file=sys.stderr)
                return 2
        if args.model is not None and args.model not in config.models:
            raise ConfigError("Unknown model profile.")
        if args.command == "chat":
            from ai_assistant.interfaces.chat import chat
            try:
                return asyncio.run(chat(config, args.model, session_id=args.session))
            except ConfigError:
                raise
            except (ImportError, OSError, ValueError):
                print("助理启动失败：请检查可选依赖及工作区。", file=sys.stderr)
                return 2
        if args.command == "model":
            from ai_assistant.llm.checks import check_model
            selected = config.default_model if args.model is None else args.model
            retries = config.models[selected].parameters.get("max_retries")
            retry_text = "沿用 SDK 默认" if retries is None else f"max_retries={retries}（SDK）"
            print(f"开始真实模型检查（可能产生费用），SDK 重试：{retry_text}；每项一次逻辑调用。", flush=True)
            results = asyncio.run(check_model(config, args.model, args.capability, args.timeout, on_result=_print_result))
            return 1 if any(item.status == "failed" for item in results) else 0
    except SessionError as exc:
        print(f"会话失败：{exc}", file=sys.stderr, flush=True)
        return 2
    except ConfigError as exc:
        print(f"检查失败：{exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("检查已取消，未开始的项目不再发送。", file=sys.stderr, flush=True)
        return 130
    names = [args.model] if args.model is not None else list(config.models)
    print("配置有效，未检查连通性。模型：" + ", ".join(ascii(name) for name in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
