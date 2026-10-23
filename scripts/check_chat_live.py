"""Explicit manual assistant acceptance; real calls may incur charges.

Run: .venv/bin/python scripts/check_chat_live.py --config assistant.toml
Use --write only in a workspace where creating/editing a file is authorized.
"""
import argparse
import asyncio

from ai_assistant.config import load_config
from ai_assistant.interfaces.chat import chat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--model')
    parser.add_argument('--session', help='Reopen an existing persistent session')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.write and not config.tools.files.enabled:
        parser.error('--write requires [tools.files] enabled=true')
    print('真实模型验收，可能产生费用；请依次手动输入：')
    print('1. 你好，介绍一下你能做什么。')
    print('2. 检索知识库中的发布要求，引用片段 ID；如无证据请说明。')
    print('3. 将刚才的要求归纳为三个要点。')
    if args.write:
        print('4. 检索发布要求，并在工作区新建 chat-acceptance.md 整理结果。')
        print('5. 读取刚才的文件，在末尾增加验收清单，保留已有内容。')
    print('6. 提交需要工具的任务，执行中按 Ctrl+C；清理后提交新请求并核实工作区。')
    print('7. 退出后用 --session 显示的 UUID 重开，追问修改；多轮输入较长内容并观察摘要计数。')
    print('8. /new 后询问之前的要求，验证上下文隔离；/exit 或 EOF 退出。')
    return asyncio.run(chat(config, args.model, session_id=args.session))


if __name__ == '__main__':
    raise SystemExit(main())
