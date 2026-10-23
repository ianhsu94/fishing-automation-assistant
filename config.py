from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Roi:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_value(cls, value: list[float] | tuple[float, ...]) -> "Roi":
        if len(value) != 4:
            raise ValueError(f"ROI 必須有四個值，收到：{value!r}")
        roi = cls(*(float(item) for item in value))
        if not (0 <= roi.x1 < roi.x2 <= 1 and 0 <= roi.y1 < roi.y2 <= 1):
            raise ValueError(f"ROI 必須是 0..1 的遞增比例座標，收到：{value!r}")
        return roi

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )


@dataclass(frozen=True)
class AdbConfig:
    executable: str
    serial: str
    command_timeout_seconds: float
    screenshot_mode: str
    stream_width: int
    stream_height: int
    stream_bitrate: int


@dataclass(frozen=True)
class VisionConfig:
    canonical_width: int
    canonical_height: int
    rois: dict[str, Roi]
    template_dir: Path
    min_title_score: float
    min_result_score: float
    min_bobber_score: float
    min_ready_green_fraction: float
    min_action_purple_fraction: float
    min_prompt_purple_fraction: float
    min_prompt_yellow_fraction: float
    min_yellow_width_fraction: float
    min_bar_track_fraction: float
    min_active_game_blue_fraction: float
    max_active_game_dark_fraction: float


@dataclass(frozen=True)
class ControlConfig:
    target_fps: float
    stable_frames: int
    timing_exit_ready_frames: int
    max_consecutive_capture_errors: int
    action_cooldown_seconds: float
    timing_click_cooldown_seconds: float
    timing_rearm_seconds: float
    timing_rearm_outside_frames: int
    min_timing_action_confidence: float
    max_timing_flash_green_fraction: float
    min_timing_success_white_fraction: float
    target_change_rearm_fraction: float
    target_margin_ratio: float
    latency_compensation_seconds: float
    tap_command_lead_seconds: float
    max_timing_schedule_seconds: float
    no_button_result_seconds: float
    min_valid_bobber_speed_per_second: float
    max_bobber_speed_per_second: float
    wait_bite_timeout_seconds: float
    timing_timeout_seconds: float
    result_timeout_seconds: float
    recovery_timeout_seconds: float
    result_auto_tap: bool
    result_hold_seconds: float
    result_tap_interval_seconds: float
    taps: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class OutputConfig:
    log_dir: Path
    debug_dir: Path
    save_debug_on_state_change: bool
    save_debug_on_action: bool
    cleanup_debug_each_round: bool


@dataclass(frozen=True)
class AppConfig:
    adb: AdbConfig
    vision: VisionConfig
    control: ControlConfig
    output: OutputConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "adb": {
        "executable": "【請填入 ADB 執行檔路徑】",
        "serial": "【請填入模擬器或 Android 裝置序號】",
        "command_timeout_seconds": 8.0,
        "screenshot_mode": "h264",
        "stream_width": 450,
        "stream_height": 800,
        "stream_bitrate": 2000000,
    },
    "vision": {
        "canonical_width": 709,
        "canonical_height": 1536,
        "template_dir": "templates",
        "rois": {
            "title": [0.24, 0.045, 0.76, 0.18],
            "action": [0.28, 0.70, 0.72, 0.91],
            "prompt": [0.18, 0.60, 0.82, 0.715],
            "timing_bar": [0.17, 0.625, 0.83, 0.745],
            "result": [0.16, 0.19, 0.84, 0.72],
        },
        "min_title_score": 0.30,
        "min_result_score": 0.42,
        "min_bobber_score": 0.38,
        "min_ready_green_fraction": 0.12,
        "min_action_purple_fraction": 0.10,
        "min_prompt_purple_fraction": 0.08,
        "min_prompt_yellow_fraction": 0.025,
        "min_yellow_width_fraction": 0.018,
        "min_bar_track_fraction": 0.055,
        "min_active_game_blue_fraction": 0.55,
        "max_active_game_dark_fraction": 0.10,
    },
    "control": {
        "target_fps": 30.0,
        "stable_frames": 1,
        "timing_exit_ready_frames": 4,
        "max_consecutive_capture_errors": 3,
        "action_cooldown_seconds": 0.18,
        "timing_click_cooldown_seconds": 0.38,
        "timing_rearm_seconds": 0.75,
        "timing_rearm_outside_frames": 3,
        "min_timing_action_confidence": 0.38,
        "max_timing_flash_green_fraction": 0.12,
        "min_timing_success_white_fraction": 0.17,
        "target_change_rearm_fraction": 0.075,
        "target_margin_ratio": 0.12,
        "latency_compensation_seconds": 0.34,
        "tap_command_lead_seconds": 0.012,
        "max_timing_schedule_seconds": 0.30,
        "no_button_result_seconds": 0.45,
        "min_valid_bobber_speed_per_second": 0.18,
        "max_bobber_speed_per_second": 0.80,
        "wait_bite_timeout_seconds": 40.0,
        "timing_timeout_seconds": 38.0,
        "result_timeout_seconds": 12.0,
        "recovery_timeout_seconds": 20.0,
        "result_auto_tap": True,
        "result_hold_seconds": 0.18,
        "result_tap_interval_seconds": 0.50,
        "taps": {
            "start": [0.50, 0.80],
            "hook": [0.50, 0.80],
            "timing": [0.50, 0.80],
            "result": [0.50, 0.80],
            "result_confirm": [0.50, 0.93],
        },
    },
    "output": {
        "log_dir": "logs",
        "debug_dir": "debug",
        "save_debug_on_state_change": False,
        "save_debug_on_action": False,
        "cleanup_debug_each_round": True,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path_obj = Path(__file__).resolve().with_name("config.json")
    else:
        path_obj = Path(path).expanduser().resolve()

    supplied: dict[str, Any] = {}
    if path_obj.exists():
        supplied = json.loads(path_obj.read_text(encoding="utf-8"))
    data = _deep_merge(DEFAULT_CONFIG, supplied)
    base_dir = path_obj.parent

    adb_data = data["adb"]
    vision_data = data["vision"]
    control_data = data["control"]
    output_data = data["output"]

    taps = {
        name: (float(point[0]), float(point[1]))
        for name, point in control_data["taps"].items()
    }
    for name, point in taps.items():
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            raise ValueError(f"點擊位置 {name} 必須使用 0..1 比例座標：{point}")

    return AppConfig(
        adb=AdbConfig(
            executable=str(adb_data["executable"]),
            serial=str(adb_data["serial"]),
            command_timeout_seconds=float(adb_data["command_timeout_seconds"]),
            screenshot_mode=str(adb_data["screenshot_mode"]).lower(),
            stream_width=int(adb_data["stream_width"]),
            stream_height=int(adb_data["stream_height"]),
            stream_bitrate=int(adb_data["stream_bitrate"]),
        ),
        vision=VisionConfig(
            canonical_width=int(vision_data["canonical_width"]),
            canonical_height=int(vision_data["canonical_height"]),
            rois={name: Roi.from_value(value) for name, value in vision_data["rois"].items()},
            template_dir=_relative_path(base_dir, str(vision_data["template_dir"])),
            min_title_score=float(vision_data["min_title_score"]),
            min_result_score=float(vision_data["min_result_score"]),
            min_bobber_score=float(vision_data["min_bobber_score"]),
            min_ready_green_fraction=float(vision_data["min_ready_green_fraction"]),
            min_action_purple_fraction=float(vision_data["min_action_purple_fraction"]),
            min_prompt_purple_fraction=float(vision_data["min_prompt_purple_fraction"]),
            min_prompt_yellow_fraction=float(vision_data["min_prompt_yellow_fraction"]),
            min_yellow_width_fraction=float(vision_data["min_yellow_width_fraction"]),
            min_bar_track_fraction=float(vision_data["min_bar_track_fraction"]),
            min_active_game_blue_fraction=float(vision_data["min_active_game_blue_fraction"]),
            max_active_game_dark_fraction=float(vision_data["max_active_game_dark_fraction"]),
        ),
        control=ControlConfig(
            target_fps=float(control_data["target_fps"]),
            stable_frames=int(control_data["stable_frames"]),
            timing_exit_ready_frames=int(control_data["timing_exit_ready_frames"]),
            max_consecutive_capture_errors=int(control_data["max_consecutive_capture_errors"]),
            action_cooldown_seconds=float(control_data["action_cooldown_seconds"]),
            timing_click_cooldown_seconds=float(control_data["timing_click_cooldown_seconds"]),
            timing_rearm_seconds=float(control_data["timing_rearm_seconds"]),
            timing_rearm_outside_frames=int(control_data["timing_rearm_outside_frames"]),
            min_timing_action_confidence=float(control_data["min_timing_action_confidence"]),
            max_timing_flash_green_fraction=float(control_data["max_timing_flash_green_fraction"]),
            min_timing_success_white_fraction=float(control_data["min_timing_success_white_fraction"]),
            target_change_rearm_fraction=float(control_data["target_change_rearm_fraction"]),
            target_margin_ratio=float(control_data["target_margin_ratio"]),
            latency_compensation_seconds=float(control_data["latency_compensation_seconds"]),
            tap_command_lead_seconds=float(control_data["tap_command_lead_seconds"]),
            max_timing_schedule_seconds=float(control_data["max_timing_schedule_seconds"]),
            no_button_result_seconds=float(control_data["no_button_result_seconds"]),
            min_valid_bobber_speed_per_second=float(control_data["min_valid_bobber_speed_per_second"]),
            max_bobber_speed_per_second=float(control_data["max_bobber_speed_per_second"]),
            wait_bite_timeout_seconds=float(control_data["wait_bite_timeout_seconds"]),
            timing_timeout_seconds=float(control_data["timing_timeout_seconds"]),
            result_timeout_seconds=float(control_data["result_timeout_seconds"]),
            recovery_timeout_seconds=float(control_data["recovery_timeout_seconds"]),
            result_auto_tap=bool(control_data["result_auto_tap"]),
            result_hold_seconds=float(control_data["result_hold_seconds"]),
            result_tap_interval_seconds=float(control_data["result_tap_interval_seconds"]),
            taps=taps,
        ),
        output=OutputConfig(
            log_dir=_relative_path(base_dir, str(output_data["log_dir"])),
            debug_dir=_relative_path(base_dir, str(output_data["debug_dir"])),
            save_debug_on_state_change=bool(output_data["save_debug_on_state_change"]),
            save_debug_on_action=bool(output_data["save_debug_on_action"]),
            cleanup_debug_each_round=bool(output_data["cleanup_debug_each_round"]),
        ),
    )
