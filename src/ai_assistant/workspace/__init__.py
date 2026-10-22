"""Bounded project file access and optimistic atomic text mutations."""

from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError

from ai_assistant.workspace.writes import WritableWorkspace

__all__ = ['ReadOnlyWorkspace', 'WritableWorkspace', 'WorkspaceError']
