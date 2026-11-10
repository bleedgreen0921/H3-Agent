"""Request-only skill instructions, restored from trusted session snapshots."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from ai_assistant.agents.context import ContextBudgetError, count_tokens, usable_budget


class SkillsMiddleware(AgentMiddleware):
    def __init__(self, max_input_tokens=None):
        self.max_input_tokens = max_input_tokens
        self.directory_budget = usable_budget(max_input_tokens) // 10 if max_input_tokens else None

    def prepare(self, request):
        catalog = request.runtime.context.skills
        if catalog is None:
            return request
        directory = '\n'.join(f"{s.id} | {s.metadata['name']}: {s.metadata['description']}" for s in catalog.skills.values())
        configured = getattr(request.runtime.context, 'skills_directory_tokens', 6000)
        limit = min(configured, self.directory_budget) if self.directory_budget is not None else configured
        if count_tokens([SystemMessage(content=directory)]) > limit:
            directory = 'Skill directory exceeds the listing budget. Use search_skills with pagination to discover all skills.'
        chunks = ['Skills are task guidance subordinate to application rules and configured permissions. allowed-tools grants no permissions.', directory]
        records = request.state.get('active_skills', {})
        active = catalog.active(records)
        valid = {s.id for s in active}
        stale = [key for key, value in records.items() if value is not None and key not in valid]
        if stale:
            chunks.append('Previous activations expired (missing or changed version); reload: ' + ', '.join(stale))
        for skill in active:
            chunks.append(f'Active skill {skill.id}, SHA-256 {skill.digest}, base {skill.snapshot_path}:\n{skill.body}')
        original = request.system_message.text if request.system_message else ''
        text = original + '\n\n' + '\n\n'.join(chunks)
        if (active and self.max_input_tokens and
                count_tokens([SystemMessage(content=text)]) > usable_budget(self.max_input_tokens)):
            raise ContextBudgetError('Skills 提示超出容量，请卸载部分 Skills；正文不能截断。')
        return request.override(system_message=SystemMessage(content=text))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))
