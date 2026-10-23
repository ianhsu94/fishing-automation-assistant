from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from config import ControlConfig, Roi
from models import ActionKind, BotAction, BotState, ScreenKind, VisionObservation


@dataclass
class MachineSnapshot:
    state: BotState
    stable_kind: ScreenKind
    stable_count: int
    predicted_bobber_x: float | None


class FishingStateMachine:
    def __init__(self, config: ControlConfig, timing_roi: Roi) -> None:
        self.config = config
        self.timing_roi = timing_roi
        self.state = BotState.UNKNOWN
        self.state_entered_at = 0.0
        self.last_action_at = -1e9
        self.last_timing_click_at = -1e9
        self.last_result_click_at = -1e9
        self.result_tap_name = "result"
        self.last_seen_kind = ScreenKind.UNKNOWN
        self.stable_count = 0
        self.previous_bobber: tuple[float, float] | None = None
        self.predicted_bobber_x: float | None = None
        self.bobber_speed = 0.0
        self.has_bobber_speed = False
        self.speed_samples: list[float] = []
        self.timing_armed = True
        self.outside_frames = 0
        self.last_clicked_target: tuple[float, float] | None = None
        self.pending_target: tuple[float, float] | None = None
        self.pending_target_frames = 0
        self.success_flash_seen = False
        self.unknown_no_button_since: float | None = None
        self.pre_click_success_white = 0.0
        self.last_timing_decision = "尚未進入計時畫面"
        self.ready_action_sent = False
        self.bite_action_sent = False

    def _set_state(self, state: BotState, now: float) -> None:
        if state != self.state:
            self.state = state
            self.state_entered_at = now
            if state == BotState.TIMING:
                self.last_timing_decision = "等待本輪計時條穩定"
            else:
                if state != BotState.RESULT:
                    self.result_tap_name = "result"
                self.previous_bobber = None
                self.predicted_bobber_x = None
                self.bobber_speed = 0.0
                self.has_bobber_speed = False
                self.speed_samples.clear()
                self.timing_armed = True
                self.outside_frames = 0
                self.last_clicked_target = None
                self.pending_target = None
                self.pending_target_frames = 0
                self.success_flash_seen = False
                self.unknown_no_button_since = None
                self.pre_click_success_white = 0.0

    def _observe_stability(self, kind: ScreenKind) -> bool:
        if kind == self.last_seen_kind:
            self.stable_count += 1
        else:
            self.last_seen_kind = kind
            self.stable_count = 1
        return self.stable_count >= self.config.stable_frames

    def _predict_bobber(
        self,
        x: float,
        now: float,
        track_left: float | None = None,
        track_right: float | None = None,
        bobber_half_width: float | None = None,
    ) -> float:
        predicted = x
        if self.previous_bobber is not None:
            previous_x, previous_time = self.previous_bobber
            delta_time = now - previous_time
            # 超過 250ms 的兩點之間可能已經反彈，不能把跨越端點的平均位移當方向。
            if 0.015 <= delta_time <= 0.25:
                speed = (x - previous_x) / delta_time
                # 遊戲的移動速率固定；保留最近的速率中位數可消除像素量化抖動，
                # 方向則永遠使用最新位移，碰到左右端點反向時不會沿用舊方向。
                speed_magnitude = abs(speed)
                if (
                    self.config.min_valid_bobber_speed_per_second
                    <= speed_magnitude
                    <= self.config.max_bobber_speed_per_second + 1e-6
                ):
                    self.speed_samples.append(abs(speed))
                    del self.speed_samples[:-7]
                    magnitude = median(self.speed_samples)
                    self.bobber_speed = magnitude if speed > 0 else -magnitude
                    self.has_bobber_speed = True
                elif 0.025 < speed_magnitude < self.config.min_valid_bobber_speed_per_second and self.speed_samples:
                    # 反彈剛好發生在兩幀之間時，表面位移會變小；只採方向，速率仍用校正值。
                    magnitude = median(self.speed_samples)
                    self.bobber_speed = magnitude if speed > 0 else -magnitude
                    self.has_bobber_speed = True
                elif speed_magnitude > self.config.max_bobber_speed_per_second:
                    # 黃區切換或特效後的座標跳動不是浮標速率，等下一個正常影格再判斷。
                    self.has_bobber_speed = False
                predicted = x + self.bobber_speed * self.config.latency_compensation_seconds
            elif delta_time > 0.25:
                # 長間隔內可能已經反彈；保留固定速率樣本，但方向必須由下一對畫面重抓。
                self.has_bobber_speed = False

        padding = max(0.035, bobber_half_width or 0.0)
        left = (track_left + padding) if track_left is not None else self.timing_roi.x1
        right = (track_right - padding) if track_right is not None else self.timing_roi.x2
        if right <= left:
            left, right = self.timing_roi.x1, self.timing_roi.x2
        # 340ms 內可能碰到軌道端點；依真實軌道範圍反射，而不是延續錯誤方向。
        for _ in range(4):
            if predicted < left:
                predicted = left + (left - predicted)
            elif predicted > right:
                predicted = right - (predicted - right)
            else:
                break
        predicted = max(left, min(right, predicted))
        self.previous_bobber = (x, now)
        self.predicted_bobber_x = predicted
        return predicted

    @staticmethod
    def _advance_bouncing(
        x: float,
        direction: int,
        speed: float,
        elapsed: float,
        left: float,
        right: float,
    ) -> tuple[float, int]:
        """以等速反彈模型推進浮標，回傳位置與推進後方向。"""
        span = right - left
        if span <= 0.0 or speed <= 0.0:
            return x, direction
        x = max(left, min(right, x))
        phase = (x - left) if direction > 0 else span + (right - x)
        phase = (phase + speed * max(0.0, elapsed)) % (2.0 * span)
        if phase <= span:
            return left + phase, 1
        return right - (phase - span), -1

    def _time_until_safe_entry(
        self,
        x: float,
        safe_left: float,
        safe_right: float,
        track_left: float,
        track_right: float,
    ) -> tuple[float, float, int] | None:
        """計算從現在起多久會第一次進入黃區；包含畫面延遲、點擊延遲與反彈。"""
        if not self.has_bobber_speed or abs(self.bobber_speed) <= 0.025:
            return None
        if track_right <= track_left:
            return None
        safe_left = max(track_left, min(track_right, safe_left))
        safe_right = max(track_left, min(track_right, safe_right))
        if safe_right < safe_left:
            return None

        speed = abs(self.bobber_speed)
        direction = 1 if self.bobber_speed > 0 else -1
        command_horizon = self.config.latency_compensation_seconds + self.config.tap_command_lead_seconds
        current_x, current_direction = self._advance_bouncing(
            x,
            direction,
            speed,
            command_horizon,
            track_left,
            track_right,
        )
        if safe_left <= current_x <= safe_right:
            return 0.0, current_x, current_direction

        if current_direction > 0:
            if current_x < safe_left:
                distance = safe_left - current_x
                entry_x = safe_left
                entry_direction = 1
            else:
                distance = (track_right - current_x) + (track_right - safe_right)
                entry_x = safe_right
                entry_direction = -1
        else:
            if current_x > safe_right:
                distance = current_x - safe_right
                entry_x = safe_right
                entry_direction = -1
            else:
                distance = (current_x - track_left) + (safe_left - track_left)
                entry_x = safe_left
                entry_direction = 1
        return distance / speed, entry_x, entry_direction

    def _confirm_post_success_target(self, left: float, right: float) -> bool:
        """命中特效後仍需兩幀一致的新黃區，避免特效殘影觸發補點。"""
        if not self.success_flash_seen:
            return False

        center = (left + right) / 2.0
        if self.pending_target is None:
            same_pending = False
        else:
            pending_center = (self.pending_target[0] + self.pending_target[1]) / 2.0
            same_pending = abs(center - pending_center) <= 0.025
        if same_pending:
            self.pending_target_frames += 1
        else:
            self.pending_target = (left, right)
            self.pending_target_frames = 1
        if self.pending_target_frames < 2:
            return False
        self.pending_target = None
        self.pending_target_frames = 0
        self.success_flash_seen = False
        return True

    def _timed_out(self, now: float) -> bool:
        elapsed = now - self.state_entered_at
        timeout = {
            BotState.WAIT_BITE: self.config.wait_bite_timeout_seconds,
            BotState.TIMING: self.config.timing_timeout_seconds,
            BotState.RESULT: self.config.result_timeout_seconds,
            BotState.RECOVERY: self.config.recovery_timeout_seconds,
        }.get(self.state)
        return timeout is not None and elapsed >= timeout

    def _dismiss_result(self, now: float, reason: str) -> BotAction | None:
        if not self.config.result_auto_tap:
            return None
        if now - self.state_entered_at < self.config.result_hold_seconds:
            return None
        if now - self.last_result_click_at < self.config.result_tap_interval_seconds:
            return None
        self.last_result_click_at = now
        self.last_action_at = now
        point = self.config.taps.get(self.result_tap_name, self.config.taps["result"])
        return BotAction(ActionKind.DISMISS, point, reason)

    def update(self, observation: VisionObservation, now: float) -> BotAction | None:
        if self.state_entered_at == 0.0:
            self.state_entered_at = now

        stable = self._observe_stability(observation.kind)
        timed_out = self._timed_out(now)
        if timed_out:
            self._set_state(BotState.RECOVERY, now)
            # 逾時後不允許同一個畫面立刻重新進入並點擊；僅接受已知的安全轉場。
            if observation.kind not in (ScreenKind.READY, ScreenKind.RESULT, ScreenKind.RESULT_CONFIRM):
                return None

        if self.state == BotState.TIMING and not self.timing_armed:
            success_white = observation.features.get("timing_success_white", 0.0)
            coloured_feedback = (
                observation.features.get("ready_green", 0.0) > self.config.max_timing_flash_green_fraction
                or (
                    success_white >= self.config.min_timing_success_white_fraction
                    and success_white >= self.pre_click_success_white + 0.018
                )
            )
            short_ready_feedback = (
                observation.kind == ScreenKind.READY
                and self.stable_count < self.config.timing_exit_ready_frames
            )
            if (coloured_feedback or short_ready_feedback) and not self.success_flash_seen:
                self.success_flash_seen = True
                self.outside_frames = 0
                self.pending_target = None
                self.pending_target_frames = 0
                self.last_timing_decision = "已看到命中回饋，等待新黃區穩定"
                return None

        # 最後一擊後有些結算版型沒有任何按鈕或可辨識標題，只會成為 UNKNOWN。
        # 僅限已送過計時點擊且計時條持續消失的情況，短暫確認後視為結算並點原圓鈕位置。
        if self.state == BotState.TIMING and observation.kind == ScreenKind.UNKNOWN and not self.timing_armed:
            if self.unknown_no_button_since is None:
                self.unknown_no_button_since = now
            missing_seconds = now - self.unknown_no_button_since
            if missing_seconds >= self.config.no_button_result_seconds:
                result_started_at = self.unknown_no_button_since
                self.result_tap_name = "result"
                self._set_state(BotState.RESULT, result_started_at)
                return self._dismiss_result(now, "計時後持續沒有任何按鈕或計時條，視為結算並點擊原本圓形按鈕位置")
            self.last_timing_decision = (
                f"計時條與按鈕已消失 {missing_seconds:.2f} 秒，確認是否為無按鈕結算頁"
            )
            return None
        if observation.kind != ScreenKind.UNKNOWN:
            self.unknown_no_button_since = None

        if not stable:
            return None

        # 計時條成功時會短暫閃綠；不能用單一綠色影格離開 TIMING 並誤送 START。
        if (
            self.state == BotState.TIMING
            and observation.kind == ScreenKind.READY
            and self.stable_count < self.config.timing_exit_ready_frames
        ):
            if not self.timing_armed:
                self.success_flash_seen = True
                self.outside_frames = 0
                self.pending_target = None
                self.pending_target_frames = 0
                self.last_timing_decision = "已看到綠色命中回饋，等待新黃區穩定"
            return None

        # 同一視覺畫面是一個事件，只能觸發一次；畫面確實離開後才重新上鎖。
        if observation.kind != ScreenKind.READY:
            self.ready_action_sent = False
        if observation.kind != ScreenKind.BITE:
            self.bite_action_sent = False
        if observation.kind == ScreenKind.RESULT:
            if self.state != BotState.RESULT:
                self.result_tap_name = "result"
            self._set_state(BotState.RESULT, now)
            return self._dismiss_result(now, "結算中，點擊原本圓形按鈕位置")

        if observation.kind == ScreenKind.RESULT_CONFIRM:
            self.result_tap_name = "result_confirm"
            self._set_state(BotState.RESULT, now)
            return self._dismiss_result(now, "偵測到開心收下按鈕，點擊按鈕中央")

        if observation.kind == ScreenKind.READY:
            if self.ready_action_sent:
                return None
            self._set_state(BotState.READY, now)
            if now - self.last_action_at >= self.config.action_cooldown_seconds:
                self.ready_action_sent = True
                self.last_action_at = now
                self._set_state(BotState.WAIT_BITE, now)
                return BotAction(ActionKind.START, self.config.taps["start"], "已確認可開始畫面")
            return None

        if observation.kind == ScreenKind.BITE:
            if self.bite_action_sent:
                return None
            if self.state in (BotState.UNKNOWN, BotState.READY, BotState.WAIT_BITE, BotState.RECOVERY):
                if now - self.last_action_at >= self.config.action_cooldown_seconds:
                    self.bite_action_sent = True
                    self.last_action_at = now
                    self._set_state(BotState.TIMING, now)
                    return BotAction(ActionKind.HOOK, self.config.taps["hook"], "已確認拉魚提示")
            return None

        if observation.kind == ScreenKind.TIMING:
            self._set_state(BotState.TIMING, now)
            bar = observation.bar
            success_white = observation.features.get("timing_success_white", 0.0)
            white_feedback = (
                not self.timing_armed
                and success_white >= self.config.min_timing_success_white_fraction
                and success_white >= self.pre_click_success_white + 0.018
            )
            # 綠色閃光或藍白 Cool/Perfect 特效都代表上一次已命中；此時禁止補點。
            if (
                observation.features.get("ready_green", 0.0) > self.config.max_timing_flash_green_fraction
                or white_feedback
            ):
                if not self.timing_armed:
                    self.success_flash_seen = True
                    self.outside_frames = 0
                    self.pending_target = None
                    self.pending_target_frames = 0
                self.last_timing_decision = "已看到命中回饋，等待新黃區"
                return None
            if not bar.valid or bar.bobber_x is None or bar.target_left is None or bar.target_right is None:
                self.last_timing_decision = "橫條或浮標資料不完整"
                return None
            if bar.confidence < self.config.min_timing_action_confidence:
                self.last_timing_decision = f"橫條信心不足 {bar.confidence:.3f}"
                return None

            predicted = self._predict_bobber(
                bar.bobber_x,
                now,
                bar.track_left,
                bar.track_right,
                bar.bobber_half_width,
            )
            width = bar.target_right - bar.target_left
            margin = width * self.config.target_margin_ratio
            safe_left = bar.target_left + margin
            safe_right = bar.target_right - margin
            moving_right = self.has_bobber_speed and self.bobber_speed > 0.025
            moving_left = self.has_bobber_speed and self.bobber_speed < -0.025
            # 先由兩張畫面取得方向與固定速率，再直接計算下一次進入安全區的時間。
            # 若畫面延遲後已錯過窄黃區，會連同端點反彈計算下一次回程，不在區外補點。
            direction = "右" if moving_right else "左" if moving_left else "未定"
            padding = max(0.035, bar.bobber_half_width or 0.0)
            track_left = (bar.track_left + padding) if bar.track_left is not None else self.timing_roi.x1
            track_right = (bar.track_right - padding) if bar.track_right is not None else self.timing_roi.x2
            if track_right <= track_left:
                track_left, track_right = self.timing_roi.x1, self.timing_roi.x2
            timing_plan = None
            if moving_right or moving_left:
                timing_plan = self._time_until_safe_entry(
                    bar.bobber_x,
                    safe_left,
                    safe_right,
                    track_left,
                    track_right,
                )

            if not self.timing_armed and self._confirm_post_success_target(bar.target_left, bar.target_right):
                self.timing_armed = True
                self.outside_frames = 0

            # 第一擊若沒有命中回饋，不可永遠鎖死同一組黃區；但也不能在同一次
            # 通過時連點。至少等待指定時間，並確認整個浮標連續數幀完全離開
            # 黃區後，才允許下一次往返重新嘗試。解除鎖定的當幀不點擊。
            retry_rearmed = False
            if not self.timing_armed and not self.success_flash_seen:
                bobber_half_width = max(0.0, bar.bobber_half_width or 0.0)
                clearance = max(0.006, width * 0.08)
                clearly_outside = (
                    bar.bobber_x + bobber_half_width < bar.target_left - clearance
                    or bar.bobber_x - bobber_half_width > bar.target_right + clearance
                )
                retry_wait_elapsed = now - self.last_timing_click_at >= self.config.timing_rearm_seconds
                if retry_wait_elapsed and clearly_outside:
                    self.outside_frames += 1
                else:
                    self.outside_frames = 0
                if self.outside_frames >= self.config.timing_rearm_outside_frames:
                    self.timing_armed = True
                    self.outside_frames = 0
                    retry_rearmed = True
                    self.last_timing_decision = "未見命中回饋，浮標已離開黃區；允許下次往返重試"
            else:
                self.outside_frames = 0

            if retry_rearmed:
                return None

            cooldown_ok = now - self.last_timing_click_at >= self.config.timing_click_cooldown_seconds
            plan_ready = (
                timing_plan is not None
                and timing_plan[0] <= self.config.max_timing_schedule_seconds
            )
            if plan_ready and cooldown_ok and self.timing_armed:
                assert timing_plan is not None
                delay_seconds, entry_x, entry_direction = timing_plan
                self.timing_armed = False
                self.last_clicked_target = (bar.target_left, bar.target_right)
                self.pending_target = None
                self.pending_target_frames = 0
                self.outside_frames = 0
                self.success_flash_seen = False
                self.pre_click_success_white = success_white
                self.last_timing_click_at = now + delay_seconds
                self.last_action_at = now + delay_seconds
                entry_direction_text = "右" if entry_direction > 0 else "左"
                self.last_timing_decision = f"已排定 {delay_seconds:.3f} 秒後進入黃區時點擊"
                return BotAction(
                    ActionKind.TIMING,
                    self.config.taps["timing"],
                    f"等速計算：目前往{direction}，計入畫面延遲與反彈後，"
                    f"{delay_seconds:.3f} 秒後往{entry_direction_text}進入安全區 "
                    f"{safe_left:.3f}..{safe_right:.3f}",
                    predicted_bobber_x=entry_x,
                    delay_seconds=delay_seconds,
                )
            self.last_timing_decision = (
                f"等待：目前/預測={bar.bobber_x:.3f}/{predicted:.3f} "
                f"安全區={safe_left:.3f}..{safe_right:.3f} 黃區={bar.target_left:.3f}..{bar.target_right:.3f} "
                f"方向={direction} 速度={self.bobber_speed:.3f}/秒 "
                f"排程={'未取得' if timing_plan is None else f'{timing_plan[0]:.3f}秒'} "
                f"信心={bar.confidence:.3f} armed={self.timing_armed}"
            )
            return None

        if observation.kind == ScreenKind.UNKNOWN and self.state == BotState.RESULT:
            return self._dismiss_result(now, "結算後仍非正常釣魚介面，點擊原本圓形按鈕位置")

        if observation.kind == ScreenKind.UNKNOWN and self.state not in (
            BotState.WAIT_BITE,
            BotState.TIMING,
            BotState.RESULT,
        ):
            self._set_state(BotState.RECOVERY, now)
        return None

    def snapshot(self) -> MachineSnapshot:
        return MachineSnapshot(
            state=self.state,
            stable_kind=self.last_seen_kind,
            stable_count=self.stable_count,
            predicted_bobber_x=self.predicted_bobber_x,
        )
