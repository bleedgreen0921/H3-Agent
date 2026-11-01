"""显式运行三轮真实 Agent 验收，可能产生模型费用。"""
import argparse
import asyncio
import json

from ai_assistant.config import load_config
from ai_assistant.llm.acceptance import accept_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        results = asyncio.run(accept_agent(load_config(args.config), args.model, args.timeout))
    except KeyboardInterrupt:
        print('验收已取消，资源清理结束。')
        return 130
    except Exception:
        print('验收失败；请检查配置、连接和模型响应，未装配生产工具。')
        return 2
    print(json.dumps(results, ensure_ascii=False))
    return 0 if len(results) == 3 and all(r['passed'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
