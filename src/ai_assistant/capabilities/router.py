"""按意图、覆盖关系、schema 预算和目录预算生成 ExposurePlan。"""
from ai_assistant.capabilities.intent import extract, query_terms
from ai_assistant.capabilities.model import ExposurePlan

READ_ACTIONS = frozenset({'read', 'summarize', 'search', 'list'})


def _skill_score(descriptor, terms, intent):
    text = (descriptor.name + ' ' + descriptor.description).casefold()
    score = sum(2 for term in terms if term in text)
    score += len(descriptor.domains & intent.domains) * 4
    score += len(descriptor.actions & intent.actions) * 3
    score += len(descriptor.objects & intent.objects) * 2
    return score


def _skill_directory(registry, intent, terms, active, budget):
    all_ids = list(registry.skill_ids)
    active = list(dict.fromkeys(active))
    if not all_ids:
        return [], active
    all_tokens = registry.skill_directory_tokens(all_ids)
    if budget is None or budget <= 0 or all_tokens <= budget:
        return all_ids, active
    ranked = sorted(
        (item for item in registry.descriptors_for(all_ids) if item.id not in active),
        key=lambda item: (-_skill_score(item, terms, intent), item.id),
    )
    selected = list(active)
    used = registry.skill_directory_tokens(selected)
    for descriptor in ranked:
        if used + descriptor.guidance_tokens > budget:
            continue
        selected.append(descriptor.id)
        used += descriptor.guidance_tokens
    return selected, active


def build_plan(registry, query, *, directory_budget=6000, active_skill_ids=()):
    intent = extract(query)
    terms = query_terms(query)
    mode = getattr(registry.settings, 'mode', 'retrieval')
    all_tool_ids = registry.all_tool_ids()
    active_skills = [name for name in dict.fromkeys(active_skill_ids)
                     if name in registry.skills]

    if mode == 'all':
        skill_ids, active_skills = _skill_directory(
            registry, intent, terms, active_skills, directory_budget)
        return ExposurePlan(
            tool_ids=all_tool_ids, skill_directory_ids=skill_ids,
            active_skill_ids=active_skills, reason='all', intent=intent,
            tool_schema_tokens=registry.schema_tokens(all_tool_ids),
            skill_directory_tokens=registry.skill_directory_tokens(skill_ids))

    core_ids = list(registry.core_ids)
    core = registry.descriptors_for(core_ids)
    covered_objects = set()
    covered_domains = set()
    covered_actions = set()
    for item in core:
        if item.side_effect != 'read':
            continue
        compatible = bool(item.actions & READ_ACTIONS)
        if not compatible:
            continue
        if item.objects & intent.objects or item.domains & intent.domains:
            covered_objects.update(item.objects)
            covered_domains.update(item.domains)
            covered_actions.update(item.actions)

    read_summary = (intent.side_effect != 'write'
                    and (not intent.actions or intent.actions <= frozenset({'read', 'summarize'})))
    candidates = []
    for name in all_tool_ids:
        if name in core_ids:
            continue
        item = registry.descriptors[name]
        if intent.side_effect != 'write' and item.side_effect == 'write':
            continue
        # 有明确领域时，不相关领域不进入候选；没有领域元数据的通用工具仍可按词项召回。
        if intent.domains and item.domains and not (item.domains & intent.domains):
            continue
        if intent.objects and item.objects and not (item.objects & intent.objects):
            continue
        if (intent.side_effect != 'write' and item.side_effect == 'read'
                and (item.objects & covered_objects or item.domains & covered_domains)
                and (read_summary or not item.actions or item.actions & covered_actions)):
            continue
        score = registry.score(item, terms, intent)
        if score > 0:
            candidates.append((score, item))
    candidates.sort(key=lambda entry: (-entry[0], entry[1].id))

    max_tokens = getattr(registry.settings, 'max_schema_tokens', None)
    if max_tokens is None:
        max_tokens = getattr(getattr(registry, 'policy', None), 'max_schema_tokens', 0)
    max_tokens = max_tokens or 0
    top_k = getattr(registry.settings, 'top_k', 8) or getattr(registry.policy, 'top_k', 8)
    selected = list(core_ids)
    used = registry.schema_tokens(selected)
    overflow_core = bool(max_tokens and used > max_tokens)
    added = 0
    for _, item in candidates:
        if added >= top_k:
            break
        if max_tokens and used + item.schema_tokens > max_tokens:
            continue
        selected.append(item.id)
        used += item.schema_tokens
        added += 1

    reason = 'all' if mode == 'all' else 'retrieval'
    if overflow_core:
        reason = 'core_over_budget'
    elif covered_objects and added == 0:
        reason = 'core_coverage'
    if not selected:
        selected = list(all_tool_ids)
        used = registry.schema_tokens(selected)
        reason = 'fallback'

    skill_ids, active_skills = _skill_directory(
        registry, intent, terms, active_skills, directory_budget)
    return ExposurePlan(
        tool_ids=selected, skill_directory_ids=skill_ids,
        active_skill_ids=active_skills, reason=reason, intent=intent,
        tool_schema_tokens=used,
        skill_directory_tokens=registry.skill_directory_tokens(skill_ids))
