from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScreenKind(str, Enum):
    READY = "ready"
    BITE = "bite"
    TIMING = "timing"
    RESULT = "result"
    RESULT_CONFIRM = "result_confirm"
    UNKNOWN = "unknown"


class BotState(str, Enum):
    UNKNOWN = "UNKNOWN"
    READY = "READY"
    WAIT_BITE = "WAIT_BITE"
    TIMING = "TIMING"
    RESULT = "RESULT"
    RECOVERY = "RECOVERY"
    STOP = "STOP"


class ActionKind(str, Enum):
    START = "start"
    HOOK = "hook"
    TIMING = "timing"
    DISMISS = "dismiss"


@dataclass(frozen=True)
class BarObservation:
    target_left: float | None = None
    target_right: float | None = None
    bobber_x: float | None = None
    bobber_score: float = 0.0
    confidence: float = 0.0
    track_left: float | None = None
    track_right: float | None = None
    bobber_half_width: float | None = None

    @property
    def valid(self) -> bool:
        return (
            self.target_left is not None
            and self.target_right is not None
            and self.bobber_x is not None
            and self.target_left < self.target_right
        )


@dataclass(frozen=True)
class VisionObservation:
    kind: ScreenKind
    confidence: float
    bar: BarObservation = field(default_factory=BarObservation)
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": round(self.confidence, 4),
            "bar": {
                "target_left": self.bar.target_left,
                "target_right": self.bar.target_right,
                "bobber_x": self.bar.bobber_x,
                "bobber_score": round(self.bar.bobber_score, 4),
                "confidence": round(self.bar.confidence, 4),
                "track_left": self.bar.track_left,
                "track_right": self.bar.track_right,
            },
            "features": {key: round(value, 4) for key, value in self.features.items()},
        }


@dataclass(frozen=True)
class BotAction:
    kind: ActionKind
    point: tuple[float, float]
    reason: str
    predicted_bobber_x: float | None = None
    delay_seconds: float = 0.0
