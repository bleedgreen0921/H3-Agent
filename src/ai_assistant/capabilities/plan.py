"""基于显式边界的计划组装、后置校验和预算裁剪。"""
from ai_assistant.capabilities.model import ExposurePlan


def _settings_value(registry, name, default=0):
    """从注册表配置读取数值预算；缺失时使用安全默认值。"""
    settings = getattr(registry, 'settings', None)
    value = getattr(settings, name, default) if settings is not None else default
    return value if value is not None else default


def _effective_effect(plan, selection):
    """显式效果优先；普通对话由 requires_write 决定本轮边界。"""
    if plan.effect in {'read', 'write'}:
        return plan.effect
    return 'write' if getattr(selection, 'requires_write', False) else 'read'


def _core_ids(registry, effect):
    """核心工具始终保留，但只读轮次仍不绑定写工具。"""
    selected = []
    for name in registry.core_ids:
        descriptor = registry.descriptors.get(name)
        if descriptor is None or not descriptor.available:
            continue
        if effect == 'read' and descriptor.write:
            continue
        selected.append(name)
    return selected


def core_plan(registry, *, effect=None, reason='core', active_skill_ids=()):
    """生成只包含核心工具的安全计划。"""
    effective = effect if effect in {'read', 'write'} else 'read'
    tools = _core_ids(registry, effective)
    active = [item for item in dict.fromkeys(active_skill_ids)
              if item in registry.skills]
    plan = ExposurePlan(
        tool_ids=tools, reason=reason, selection_source='core',
        active_skill_ids=active, skill_directory_ids=list(active),
        skill_directory_tokens=registry.skill_directory_tokens(active),
        effect=effective, max_schema_tokens=_settings_value(
            registry, 'max_schema_tokens', 0),
        tool_schema_tokens=registry.schema_tokens(tools))
    return plan


def _pack_skills(registry, selected_ids, active_ids, budget):
    """已激活 Skill 无条件保留；其余目录项按顺序尝试放入剩余预算。"""
    chosen = []
    for skill_id in active_ids:
        if skill_id in registry.skills and skill_id not in chosen:
            chosen.append(skill_id)
    used = registry.skill_directory_tokens(chosen)
    for skill_id in selected_ids:
        if skill_id in chosen or skill_id not in registry.skills:
            continue
        descriptor = registry.descriptors.get(skill_id)
        cost = descriptor.guidance_tokens if descriptor is not None else 0
        if budget and used + cost > budget:
            continue
        chosen.append(skill_id)
        used += cost
    return chosen, used


def llm_plan(registry, selection, *, eligible_tools, eligible_skills,
             directory_budget=6000, active_skill_ids=(), effect=None,
             server=None, identity=None, forbid_create=False, reason='llm'):
    """在 LLM 选择结果上执行候选校验、效果过滤和预算裁剪。"""
    plan = ExposurePlan(
        reason=reason, selection_source='llm',
        candidate_tool_count=len(eligible_tools),
        candidate_skill_count=len(eligible_skills),
        requires_write=bool(getattr(selection, 'requires_write', False)),
        server=server, identity=identity, forbid_create=forbid_create,
        max_schema_tokens=_settings_value(registry, 'max_schema_tokens', 0))
    plan.effect = effect if effect in {'read', 'write'} else _effective_effect(
        plan, selection)
    plan.requires_write = plan.effect == 'write'

    # 先计入核心工具，再按选择顺序添加额外工具。
    selected = _core_ids(registry, plan.effect)
    used = registry.schema_tokens(selected)
    maximum = int(plan.max_schema_tokens or 0)
    top_k = max(1, int(_settings_value(registry, 'top_k', 8) or 8))
    candidates = {item.id: item for item in eligible_tools}
    added = 0
    for tool_id in getattr(selection, 'tool_ids', ()) or ():
        if added >= top_k:
            break
        descriptor = candidates.get(tool_id)
        if descriptor is None or descriptor.id in selected:
            continue
        if not _allowed(descriptor, plan):
            continue
        if maximum and used + descriptor.schema_tokens > maximum:
            continue
        selected.append(descriptor.id)
        used += descriptor.schema_tokens
        added += 1

    plan.tool_ids = selected
    plan.tool_schema_tokens = used

    # Skill 选择只影响目录；选择目录项不等于激活 Skill。
    active = [item for item in active_skill_ids if item in registry.skills]
    skill_candidates = {item.id for item in eligible_skills}
    selected_skills = [item for item in (getattr(selection, 'skill_ids', ()) or ())
                       if item in skill_candidates]
    directory, directory_tokens = _pack_skills(
        registry, selected_skills, active, directory_budget)
    plan.active_skill_ids = active
    plan.skill_directory_ids = directory
    plan.skill_directory_tokens = directory_tokens
    return plan


def _allowed(descriptor, plan):
    """后置校验只做断言，不扩大入口候选集。"""
    if not descriptor.available:
        return False
    if descriptor.side_effect not in {'read', 'write'}:
        return False
    if plan.effect == 'read' and descriptor.write:
        return False
    if plan.forbid_create and descriptor.creates:
        return False
    if descriptor.server is not None:
        if plan.server is not None and descriptor.server != plan.server:
            return False
        if plan.identity is not None and descriptor.identity != plan.identity:
            return False
    return True


def extend_tools(registry, plan, selected_ids, *, eligible_tools, limit=None):
    """按当前计划约束扩展工具；返回新增描述、预算跳过数和耗尽标记。"""
    candidates = {item.id: item for item in eligible_tools}
    maximum = int(plan.max_schema_tokens or _settings_value(
        registry, 'max_schema_tokens', 0) or 0)
    top_k = max(1, int(limit or _settings_value(registry, 'top_k', 8) or 8))
    used = plan.tool_schema_tokens or registry.schema_tokens(plan.tool_ids)
    added, skipped = [], 0
    for tool_id in selected_ids:
        if len(added) >= top_k:
            break
        descriptor = candidates.get(tool_id)
        if descriptor is None or tool_id in plan.tool_ids:
            continue
        if not _allowed(descriptor, plan):
            continue
        if maximum and used + descriptor.schema_tokens > maximum:
            skipped += 1
            continue
        plan.tool_ids.append(descriptor.id)
        added.append(descriptor)
        used += descriptor.schema_tokens
    plan.tool_schema_tokens = used
    return added, skipped, bool(skipped and not added)


def extend_skills(registry, plan, selected_ids, *, eligible_skills,
                  directory_budget=6000):
    """只扩展 Skill 目录，不激活正文。"""
    candidates = {item.id: item for item in eligible_skills}
    used = plan.skill_directory_tokens or registry.skill_directory_tokens(
        plan.skill_directory_ids)
    added, skipped = [], 0
    for skill_id in selected_ids:
        descriptor = candidates.get(skill_id)
        if descriptor is None or skill_id in plan.skill_directory_ids:
            continue
        if directory_budget and used + descriptor.guidance_tokens > directory_budget:
            skipped += 1
            continue
        plan.skill_directory_ids.append(descriptor.id)
        added.append(descriptor)
        used += descriptor.guidance_tokens
    plan.skill_directory_tokens = used
    return added, skipped, bool(skipped and not added)
