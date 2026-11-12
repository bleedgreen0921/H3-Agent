"""渠道抽象；不包含飞书字段或 Markdown 协议。"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class InboundEnvelope:
    id: str
    binding: str
    chat_id: str
    text: str
    received_at: object = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundPart:
    id: str
    input_id: str
    part: int
    payload: dict
    delivery: dict = field(default_factory=dict)
    presentation_ref: str | None = None


@dataclass(frozen=True)
class DeliveryPolicy:
    max_attempts: int
    final_timeout_seconds: float
    attempt_timeout_seconds: float


class ProgressSink(Protocol):
    def start(self): ...
    def event(self, item): ...
    async def finish(self, state, *, phase=None): ...


class Channel(Protocol):
    async def receive(self, envelope): ...
    async def deliver(self, part: OutboundPart): ...
