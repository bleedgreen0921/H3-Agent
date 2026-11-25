"""启动本地 Web 服务。"""
import asyncio


def _preflight(config):
    from ai_assistant.config import ConfigError
    from ai_assistant.storage.attachments import attachment_store
    from ai_assistant.storage.sessions import session_storage
    from ai_assistant.storage.web import web_store
    async def check():
        try:
            async with session_storage(config.sessions):
                pass
            async with web_store(config.sessions):
                pass
            if config.attachments.enabled:
                async with attachment_store(config.sessions):
                    pass
        except Exception:
            raise ConfigError("Web 启动前置检查失败：请确认数据库可连接并已显式执行 init。") from None
    asyncio.run(check())


def serve(config):
    import uvicorn
    from ai_assistant.web.app import create_app

    _preflight(config)
    app = create_app(config)
    token = config.web.token_path()
    host, port = config.web.host, config.web.port
    print(f"Web 服务：http://{host}:{port}/?token={app.state.token}", flush=True)
    print(f"令牌文件：{token}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
