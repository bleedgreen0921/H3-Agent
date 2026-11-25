"""删除旧渠道绑定会话及其展示/投递数据；默认预览，知识库保留。"""
import argparse
import asyncio
import json

from ai_assistant.config import ConfigError, load_config

TABLES = (
    "assistant_outbox", "assistant_inbox", "assistant_chat_commands",
    "assistant_channel_sessions", "assistant_operations",
    "assistant_model_usage", "assistant_attachments",
    "checkpoint_writes", "checkpoint_blobs", "checkpoints",
    "assistant_sessions",
)


async def run(config, *, execute=False):
    if config.sessions is None:
        raise ConfigError("未配置会话存储。")
    import psycopg
    from psycopg.rows import dict_row
    from ai_assistant.storage.sessions import lock_key

    conn = await psycopg.AsyncConnection.connect(
        config.sessions.database_url.get_secret_value(), autocommit=False,
        row_factory=dict_row)
    async with conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL lock_timeout='2s'")
            await conn.execute("SET LOCAL statement_timeout='30s'")
            # 只删除具有渠道绑定的旧会话；Web 会话 binding 为空。
            rows = await (await conn.execute(
                "SELECT id, binding, thread_ids, status FROM assistant_sessions "
                "WHERE binding IS NOT NULL ORDER BY updated_at")).fetchall()
            if not rows:
                return {"mode": "executed" if execute else "preview",
                        "sessions": 0, "knowledge": "untouched"}
            if not execute:
                return {"mode": "preview", "sessions": len(rows),
                        "sample": [dict(row) for row in rows[:5]],
                        "knowledge": "untouched"}
            ids = [row["id"] for row in rows]
            threads = [thread for row in rows for thread in (row["thread_ids"] or [])]
            bindings = [row["binding"] for row in rows]

            async def exists(name):
                value = await (await conn.execute(
                    "SELECT to_regclass(%s) AS name", (name,))).fetchone()
                return bool(value and value["name"])

            for ident in ids:
                await conn.execute("SELECT pg_advisory_xact_lock(%s)",
                                   (lock_key(str(ident)),))
            if threads:
                if await exists("checkpoint_writes"):
                    await conn.execute(
                        "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)",
                        (threads,))
                if await exists("checkpoint_blobs"):
                    await conn.execute(
                        "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)",
                        (threads,))
                if await exists("checkpoints"):
                    await conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ANY(%s)",
                        (threads,))
            if await exists("assistant_outbox") and await exists("assistant_inbox"):
                await conn.execute(
                    "DELETE FROM assistant_outbox WHERE input_id IN "
                    "(SELECT id FROM assistant_inbox WHERE binding = ANY(%s))",
                    (bindings,))
            if await exists("assistant_operations") and await exists("assistant_inbox"):
                await conn.execute(
                    "DELETE FROM assistant_operations WHERE input_id IN "
                    "(SELECT id FROM assistant_inbox WHERE binding = ANY(%s))",
                    (bindings,))
            if await exists("assistant_inbox"):
                await conn.execute(
                    "DELETE FROM assistant_inbox WHERE binding = ANY(%s)",
                    (bindings,))
            if await exists("assistant_chat_commands"):
                await conn.execute(
                    "DELETE FROM assistant_chat_commands WHERE binding = ANY(%s)",
                    (bindings,))
            if await exists("assistant_attachments"):
                await conn.execute(
                    "DELETE FROM assistant_attachments WHERE session_id = ANY(%s)",
                    (ids,))
            if await exists("assistant_web_messages"):
                await conn.execute(
                    "DELETE FROM assistant_web_messages WHERE session_id = ANY(%s)",
                    (ids,))
            if await exists("assistant_model_usage"):
                await conn.execute(
                    "DELETE FROM assistant_model_usage WHERE session_id = ANY(%s)",
                    (ids,))
            if await exists("assistant_channel_sessions"):
                await conn.execute(
                    "DELETE FROM assistant_channel_sessions WHERE session_id = ANY(%s)",
                    (ids,))
            await conn.execute(
                "DELETE FROM assistant_sessions WHERE id = ANY(%s)", (ids,))
            return {"mode": "executed", "sessions": len(rows),
                    "knowledge": "untouched", "remote": "untouched"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="assistant.toml")
    parser.add_argument("--profile")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, profile=args.profile)
        result = asyncio.run(run(config, execute=args.execute))
    except (ConfigError, Exception) as error:
        if isinstance(error, ConfigError):
            print(str(error))
        else:
            print("删除未完成：请检查数据库连接和会话锁；已回滚。")
        return 2
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
