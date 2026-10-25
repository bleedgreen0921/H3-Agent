"""Admission of user-submitted links and trusted incoming file references."""

import json
import re
from pathlib import Path
import httpx
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.sync import register


async def _admit_message(config, text, *, attachment=False):
    results = []
    if attachment and text.startswith("File reference: "):
        from ai_assistant.integrations.feishu_files import FileBridge, FeishuAPI
        from ai_assistant.workspace.files import ReadOnlyWorkspace

        info = json.loads(text[len("File reference: ") :])
        base = config.config_path.parent if config.config_path else Path.cwd()
        root = Path(config.workspace.root)
        root = root if root.is_absolute() else base / root
        workspace = ReadOnlyWorkspace(
            root,
            config_path=config.config_path,
            exclude_patterns=config.workspace.exclude_patterns,
        )
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            result = await FileBridge(
                FeishuAPI(config.integrations.feishu, client), workspace
            ).download_message_file(
                info["message_id"], info["file_key"], info["file_name"]
            )
        results.append(await register(config, result["path"]))
    for value in dict.fromkeys(re.findall(r'https://[^\s<>"\]]+', text)):
        if re.search(
            r"\.(?:feishu\.cn|larksuite\.com)/(?:wiki|docx|sheets|base)/", value
        ):
            try:
                results.append(await register(config, value.rstrip("。，；）)")))
            except (KnowledgeError, ValueError, httpx.HTTPError) as exc:
                results.append(
                    {
                        "status": "failed",
                        "reason": getattr(exc, "code", type(exc).__name__),
                    }
                )
    return results


async def admit_message(config, text, *, attachment=False):
    """登记失败作为观察结果交给 Agent，不阻断本轮正常问答。"""
    try:
        return await _admit_message(config, text, attachment=attachment)
    except (KnowledgeError, ValueError, KeyError, httpx.HTTPError) as exc:
        return [
            {"status": "failed", "reason": getattr(exc, "code", type(exc).__name__)}
        ]
