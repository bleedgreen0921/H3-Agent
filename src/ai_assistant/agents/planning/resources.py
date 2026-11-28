"""资源引用提取、稳定绑定和工具种子选择。"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from ai_assistant.agents.planning.models import (
    PlanValidationError, ResourceRef, TaskExecutionPlan,
)
from ai_assistant.capabilities.plan import core_plan

_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
_FEISHU_HOSTS = ('feishu.cn', 'larksuite.com', 'feishu.net')
_RESOURCE_KINDS = {
    'feishu_wiki': 'document', 'feishu_doc': 'document',
    'feishu_sheet': 'document', 'feishu_base': 'record',
    'file': 'file',
}


def _normalize_url(value):
    return value.rstrip('.,;:!?)]}>') if value else value


def _extract_resources(text, start=0):
    """从用户原文提取稳定资源；URL 只保留 token，不依赖后续模型复制。"""
    resources, seen = [], set()
    for match in _URL.findall(text or ''):
        url = _normalize_url(match)
        parsed = urlparse(url)
        host = (parsed.hostname or '').casefold()
        if not any(part in host for part in _FEISHU_HOSTS):
            continue
        parts = [part for part in parsed.path.split('/') if part]
        kind, token = 'other', None
        for index, part in enumerate(parts):
            if part in {'wiki', 'docx', 'docs', 'sheets', 'base'} and index + 1 < len(parts):
                kind = {'wiki': 'feishu_wiki', 'docx': 'feishu_doc', 'docs': 'feishu_doc',
                        'sheets': 'feishu_sheet', 'base': 'feishu_base'}[part]
                token = parts[index + 1]
                break
        if kind == 'other':
            continue
        key = token or url
        if key in seen:
            continue
        seen.add(key)
        resources.append(ResourceRef(id=f'r{start + len(resources) + 1}', kind=kind,
                                     url=url, token=token, label=token))
    return resources


def _merge_resources(resources, text):
    existing = list(resources or [])
    keys = {(item.get('token') or item.get('url')) if isinstance(item, dict)
            else (item.token or item.url) for item in existing}
    for item in _extract_resources(text, start=len(existing)):
        key = item.token or item.url
        if key in keys:
            continue
        keys.add(key)
        existing.append(item.model_dump(mode='json'))
    return existing


def _resource_map(planning):
    return {item['id']: ResourceRef.model_validate(item)
            for item in (planning or {}).get('resources') or []}


def _bind_resources(plan, resources):
    """Planner 没写 resources 时补绑；也把 URL/token 归一成资源 ID。"""
    known = {item['id'] for item in resources}
    aliases = {}
    for item in resources:
        for value in (item.get('url'), item.get('token'), item.get('label')):
            if value:
                aliases[str(value)] = item['id']
    for task in plan.tasks:
        if task.resources:
            normalized = []
            for item in task.resources:
                resource_id = item if item in known else aliases.get(str(item))
                if resource_id is None:
                    raise PlanValidationError('任务引用了不存在的资源 ID。')
                normalized.append(resource_id)
            task.resources = list(dict.fromkeys(normalized))
            continue
        if (task.effect == 'write' and not task.creates
                and len(resources) == 1 and _feishu_task(task)):
            task.resources = [resources[0]['id']]
    return plan


def _resource_kinds(refs):
    """把资源引用映射为显式资源类型；未知类型不参与工具种子。"""
    return {_RESOURCE_KINDS[item.kind] for item in refs
            if item.kind in _RESOURCE_KINDS}


def _resolve_resource(resource, registry, default_identity):
    kind = _RESOURCE_KINDS.get(resource.kind)
    if kind is None:
        return resource
    candidates = [item for item in registry.descriptors.values()
                  if item.kind == 'tool' and item.resource_kinds & {kind}]
    preferred = [item for item in candidates
                 if not item.identity or item.identity == default_identity]
    if not preferred:
        raise PlanValidationError(
            f'没有可用的 {default_identity} 资源工具身份；'
            '请检查资源工具元数据。')
    servers = sorted({item.server for item in preferred if item.server})
    if not servers:
        return resource
    return resource.model_copy(update={'server': servers[0], 'identity': default_identity})


def _feishu_task(task):
    text = f"{task.description} {task.intent}".casefold()
    return any(word in text for word in ('飞书', '云文档', 'wiki', 'docx', '文档'))


def _resolve_task(task, planning, registry, settings):
    resources = _resource_map(planning)
    refs = []
    for resource_id in task.resources:
        if resource_id not in resources:
            raise PlanValidationError('任务引用的资源不存在。')
        refs.append(_resolve_resource(resources[resource_id], registry,
                                      settings.default_document_identity))
    effect = task.effect
    if effect == 'write' and not refs and not task.creates:
        raise PlanValidationError('写任务缺少目标资源；请补充文档链接或明确创建新文档。')
    identities = {(item.server, item.identity) for item in refs
                  if item.server or item.identity}
    if len(identities) > 1:
        raise PlanValidationError('同一任务不能混用多个飞书身份或服务。')
    server, identity = next(iter(identities), (None, None))
    if refs and not identity and any(item.server or item.identity for item in refs):
        raise PlanValidationError('无法确定目标资源的飞书身份。')
    if task.creates and not identity and _feishu_task(task):
        resolved = _resolve_resource(
            ResourceRef(id='r1', kind='feishu_doc'), registry,
            settings.default_document_identity)
        server, identity = resolved.server, resolved.identity

    resource_kinds = _resource_kinds(refs)
    plan = core_plan(registry, effect=effect)
    if refs and resource_kinds:
        # 目标资源的读写工具由显式 resource_kinds 决定，不经过关键词路由。
        try:
            plan = registry.seed_resource_tools(
                plan, resource_kinds=resource_kinds, server=server,
                identity=identity, effect=effect,
                forbid_create=not task.creates, append=task.append)
        except ValueError as error:
            raise PlanValidationError(str(error)) from None
    if task.creates:
        kinds = resource_kinds or ({'document'} if _feishu_task(task) else {'file'})
        try:
            plan = registry.seed_create_tools(
                plan, resource_kinds=kinds, server=server, identity=identity)
        except ValueError as error:
            raise PlanValidationError(str(error)) from None
    if refs and resource_kinds and not any(
            item.resource_kinds & resource_kinds
            for name in plan.tool_ids
            if (item := registry.descriptors.get(name)) is not None):
        raise PlanValidationError(
            '目标资源没有显式声明的可用工具；'
            '请检查资源工具元数据。')
    plan.effect = effect
    plan.server = server
    plan.identity = identity
    plan.forbid_create = bool(refs and not task.creates)
    return TaskExecutionPlan(
        task_id=task.id, effect=effect, tool_ids=list(plan.tool_ids),
        schema_tokens=plan.tool_schema_tokens, resources=refs,
        server=server, identity=identity,
        forbid_create=bool(refs and not task.creates), reason='resolved')


def _resolve_plan(planning, plan, registry, settings):
    resolved = {}
    for task in plan.tasks:
        resolved[task.id] = _resolve_task(task, planning, registry, settings).model_dump(mode='json')
    return {**planning, 'resolved': resolved}
