from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from config import load_config  # noqa: E402
from models import ActionKind, BarObservation, ScreenKind, VisionObservation  # noqa: E402
from state_machine import FishingStateMachine  # noqa: E402


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(PROJECT / "config.json")
        self.machine = FishingStateMachine(config.control, config.vision.rois["timing_bar"])

    def stable(self, observation: VisionObservation, start: float):
        first = self.machine.update(observation, start)
        return first if first is not None else self.machine.update(observation, start + 0.4)

    def predicted_click(
        self,
        left: float,
        right: float,
        now: float,
        *,
        direction: int = 1,
        speed: float = 0.35,
        features: dict[str, float] | None = None,
    ):
        desired = (left + right) / 2.0
        current = desired - direction * speed * self.machine.config.latency_compensation_seconds
        previous = current - direction * speed * 0.10
        before = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(left, right, previous, 0.9, 0.9),
            features or {},
        )
        current_observation = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(left, right, current, 0.9, 0.9),
            features or {},
        )
        self.machine.update(before, now - 0.10)
        return self.machine.update(current_observation, now), current

    def test_ready_then_bite(self) -> None:
        ready = VisionObservation(ScreenKind.READY, 0.9)
        action = self.stable(ready, 10.0)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.START, action.kind)
        self.assertIsNone(self.machine.update(ready, 11.5), "同一個 READY 畫面不可重複點擊")

        bite = VisionObservation(ScreenKind.BITE, 0.9)
        action = self.stable(bite, 12.0)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.HOOK, action.kind)
        self.assertIsNone(self.machine.update(bite, 14.0), "同一個 BITE 畫面不可重複點擊")

    def test_timing_click_only_inside_target(self) -> None:
        outside = VisionObservation(
            ScreenKind.TIMING,
            0.8,
            BarObservation(target_left=0.45, target_right=0.60, bobber_x=0.30, bobber_score=0.8, confidence=0.8),
        )
        self.assertIsNone(self.stable(outside, 20.0))
        action, _ = self.predicted_click(0.45, 0.60, 21.0)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.TIMING, action.kind)
        self.assertIsNone(
            self.machine.update(
                VisionObservation(ScreenKind.TIMING, 0.8, BarObservation(0.45, 0.60, 0.52, 0.8, 0.8)),
                21.8,
            ),
            "黃色區未改變且浮標未離開時不可重複點擊",
        )

    def test_timing_clicks_early_when_predicted_position_enters_target(self) -> None:
        action, current = self.predicted_click(0.45, 0.60, 20.10, speed=0.40)
        self.assertIsNotNone(action, "目前仍在黃區外、但預測即將進入時應提前點擊")
        self.assertLess(current, 0.45)
        self.assertEqual(ActionKind.TIMING, action.kind)

    def test_timing_predicts_leftward_entry_separately(self) -> None:
        action, current = self.predicted_click(0.45, 0.60, 25.10, direction=-1, speed=0.40)
        self.assertIsNotNone(action, "往左移動時應以負速度預測進入安全區")
        self.assertLess(action.predicted_bobber_x, current)

    def test_direction_reversal_immediately_discards_old_direction(self) -> None:
        self.machine._predict_bobber(0.30, 26.0)
        self.machine._predict_bobber(0.37, 26.1)
        self.machine._predict_bobber(0.35, 26.2)
        self.assertLess(self.machine.bobber_speed, 0.0, "方向反轉後不可殘留舊的右移速度")

    def test_timing_allows_safe_current_position_with_prediction_near_yellow_edge(self) -> None:
        action, current = self.predicted_click(0.45, 0.60, 40.10, speed=0.20)
        self.assertIsNotNone(action)
        self.assertGreaterEqual(current, 0.45)
        self.assertEqual(ActionKind.TIMING, action.kind)

    def test_timing_changed_target_still_respects_hard_cooldown(self) -> None:
        action, _ = self.predicted_click(0.40, 0.60, 50.0)
        self.assertIsNotNone(action)

        flash = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.40, 0.60, 0.50, 0.9, 0.9),
            {"ready_green": 0.14},
        )
        self.assertIsNone(self.machine.update(flash, 50.05))

        changed = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.65, 0.80, 0.64, 0.9, 0.9),
        )
        self.assertIsNone(self.machine.update(changed, 50.10))
        changed_again = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.65, 0.80, 0.65, 0.9, 0.9),
        )
        self.assertIsNone(self.machine.update(changed_again, 50.15), "黃區已換位也不可在 150ms 內連點")

    def test_missed_target_does_not_rearm_before_delay(self) -> None:
        action, _ = self.predicted_click(0.40, 0.60, 60.0)
        self.assertIsNotNone(action)

        outside = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.40, 0.60, 0.25, 0.9, 0.9),
        )
        for timestamp in (60.40, 60.50, 60.60, 60.70):
            self.assertIsNone(self.machine.update(outside, timestamp))
        self.assertFalse(self.machine.timing_armed, "未滿 0.75 秒時不可解除鎖定")

    def test_missed_target_rearms_only_after_three_outside_frames(self) -> None:
        action, _ = self.predicted_click(0.40, 0.60, 60.0)
        self.assertIsNotNone(action)

        outside = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.40, 0.60, 0.25, 0.9, 0.9),
        )
        self.assertIsNone(self.machine.update(outside, 60.76))
        self.assertIsNone(self.machine.update(outside, 60.80))
        self.assertFalse(self.machine.timing_armed, "只離開兩幀時不可解除鎖定")
        self.assertIsNone(self.machine.update(outside, 60.84), "解除鎖定當幀不可立刻補點")
        self.assertTrue(self.machine.timing_armed, "等待完成且連續離開三幀後應允許下一次往返重試")

        # 模擬下一次往返：仍須先判斷方向，且延遲後的落點在安全區才可再點。
        self.machine.previous_bobber = None
        self.machine.has_bobber_speed = False
        retry, _ = self.predicted_click(0.40, 0.60, 61.10)
        self.assertIsNotNone(retry)
        self.assertEqual(ActionKind.TIMING, retry.kind)

    def test_missed_target_never_rearms_while_bobber_remains_in_target(self) -> None:
        action, _ = self.predicted_click(0.40, 0.60, 65.0)
        self.assertIsNotNone(action)

        still_inside = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.40, 0.60, 0.50, 0.9, 0.9),
        )
        for timestamp in (65.80, 65.90, 66.00, 66.10):
            self.assertIsNone(self.machine.update(still_inside, timestamp))
        self.assertFalse(self.machine.timing_armed, "浮標未完全離開黃區時不得補點")

    def test_success_flash_rearms_even_when_new_target_overlaps(self) -> None:
        action, _ = self.predicted_click(0.35, 0.50, 70.0)
        self.assertIsNotNone(action)

        flash = VisionObservation(ScreenKind.READY, 0.9)
        self.assertIsNone(self.machine.update(flash, 70.10))

        speed = 0.30
        desired = (0.37 + 0.52) / 2.0
        current = desired - speed * self.machine.config.latency_compensation_seconds
        moved = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.37, 0.52, current - speed * 0.04, 0.9, 0.9),
        )
        self.assertIsNone(self.machine.update(moved, 71.0), "新黃色區必須連續兩幀一致")
        moved_again = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.37, 0.52, current, 0.9, 0.9),
        )
        action = self.machine.update(moved_again, 71.04)
        self.assertIsNotNone(action, "成功回饋後，即使新舊黃區重疊也應允許下一次點擊")
        self.assertEqual(ActionKind.TIMING, action.kind)
        self.assertIsNone(self.machine.update(moved_again, 71.5), "新黃色區段也只能點一次")

    def test_predicted_endpoint_can_hit_narrow_target(self) -> None:
        action, _ = self.predicted_click(0.45, 0.48, 80.10, speed=0.35)
        self.assertIsNotNone(action, "延遲後的預測落點在窄黃區內時應點擊")
        self.assertGreaterEqual(action.predicted_bobber_x, 0.45)
        self.assertLessEqual(action.predicted_bobber_x, 0.48)

    def test_narrow_target_crossed_during_latency_is_scheduled_for_next_return(self) -> None:
        before = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.55, 0.56, 0.59, 0.9, 0.9, 0.17, 0.83, 0.0),
        )
        crossing = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.55, 0.56, 0.65, 0.9, 0.9, 0.17, 0.83, 0.0),
        )
        self.assertIsNone(self.machine.update(before, 81.0))
        action = self.machine.update(crossing, 81.10)
        self.assertIsNotNone(action, "即使一幀跨過極窄黃區，也應算出下一次回程進入時間")
        self.assertGreater(action.delay_seconds, 0.0, "已經錯過時不可在區外立刻點")
        self.assertGreater(self.machine.predicted_bobber_x, 0.56, "一般逐幀預測確實已越過窄黃區")
        self.assertGreaterEqual(action.predicted_bobber_x, 0.55)
        self.assertLessEqual(action.predicted_bobber_x, 0.56)

    def test_long_timing_plan_keeps_recalculating_instead_of_locking_early(self) -> None:
        before = VisionObservation(ScreenKind.TIMING, 0.9, BarObservation(0.55, 0.56, 0.59, 0.9, 0.9))
        crossing = VisionObservation(ScreenKind.TIMING, 0.9, BarObservation(0.55, 0.56, 0.65, 0.9, 0.9))
        self.assertIsNone(self.machine.update(before, 82.0))
        self.assertIsNone(self.machine.update(crossing, 82.10))
        self.assertTrue(self.machine.timing_armed, "超過 0.30 秒的遠距離預測不可提早鎖死")
        self.assertIn("0.400秒", self.machine.last_timing_decision)

    def test_no_button_screen_after_timing_is_dismissed_as_result(self) -> None:
        action, _ = self.predicted_click(0.35, 0.55, 85.0)
        self.assertIsNotNone(action)

        no_buttons = VisionObservation(ScreenKind.UNKNOWN, 0.0)
        self.assertIsNone(self.machine.update(no_buttons, 85.10))
        self.assertIsNone(self.machine.update(no_buttons, 85.50), "未滿確認時間前不可誤點短暫轉場")
        dismiss = self.machine.update(no_buttons, 85.56)
        self.assertIsNotNone(dismiss)
        self.assertEqual(ActionKind.DISMISS, dismiss.kind)
        self.assertEqual((0.50, 0.80), dismiss.point)
        self.assertEqual("RESULT", self.machine.state.value)

    def test_unknown_without_a_prior_timing_click_is_not_dismissed(self) -> None:
        timing = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(0.35, 0.55, 0.30, 0.9, 0.9),
        )
        self.assertIsNone(self.machine.update(timing, 86.0))
        unknown = VisionObservation(ScreenKind.UNKNOWN, 0.0)
        self.assertIsNone(self.machine.update(unknown, 86.1))
        self.assertIsNone(self.machine.update(unknown, 87.0), "尚未送出計時點擊時不可把未知畫面當結算")

    def test_cool_white_feedback_marks_previous_click_successful(self) -> None:
        action, _ = self.predicted_click(
            0.35,
            0.55,
            90.10,
            features={"timing_success_white": 0.14},
        )
        self.assertIsNotNone(action)

        cool = VisionObservation(
            ScreenKind.UNKNOWN,
            0.9,
            BarObservation(),
            {"timing_success_white": 0.18},
        )
        self.assertIsNone(self.machine.update(cool, 90.20))
        self.assertTrue(self.machine.success_flash_seen)

    def test_timing_green_flash_never_clicks(self) -> None:
        bar = BarObservation(
            target_left=0.40,
            target_right=0.60,
            bobber_x=0.50,
            bobber_score=0.9,
            confidence=0.9,
        )
        flash = VisionObservation(ScreenKind.TIMING, 0.9, bar, {"ready_green": 0.14})
        self.assertIsNone(self.machine.update(flash, 30.0))

    def test_timeout_does_not_reenter_same_screen(self) -> None:
        timing = VisionObservation(
            ScreenKind.TIMING,
            0.8,
            BarObservation(target_left=0.45, target_right=0.60, bobber_x=0.52, bobber_score=0.8, confidence=0.8),
        )
        self.stable(timing, 30.0)
        action = self.machine.update(timing, 80.0)
        self.assertIsNone(action)
        self.assertEqual("RECOVERY", self.machine.state.value)

    def test_timing_ignores_short_ready_flash(self) -> None:
        timing = VisionObservation(
            ScreenKind.TIMING,
            0.9,
            BarObservation(target_left=0.45, target_right=0.60, bobber_x=0.30, bobber_score=0.9, confidence=0.9),
        )
        self.stable(timing, 10.0)
        ready = VisionObservation(ScreenKind.READY, 0.4)
        for index in range(self.machine.config.timing_exit_ready_frames - 1):
            self.assertIsNone(self.machine.update(ready, 11.0 + index * 0.04))
            self.assertEqual("TIMING", self.machine.state.value)

    def test_result_and_unknown_dialogs_use_circle_position(self) -> None:
        result = VisionObservation(ScreenKind.RESULT, 0.9)
        self.assertIsNone(self.machine.update(result, 10.0))
        action = self.machine.update(result, 10.25)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.DISMISS, action.kind)
        self.assertEqual((0.50, 0.80), action.point)

        unknown = VisionObservation(ScreenKind.UNKNOWN, 0.1)
        self.assertIsNone(self.machine.update(unknown, 10.30))
        self.assertEqual("RESULT", self.machine.state.value, "結算動畫中的短暫未知影格不可進 RECOVERY")
        action = self.machine.update(unknown, 10.80)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.DISMISS, action.kind)
        self.assertEqual((0.50, 0.80), action.point)

        confirm = VisionObservation(ScreenKind.RESULT_CONFIRM, 0.9)
        action = self.machine.update(confirm, 11.35)
        self.assertIsNotNone(action)
        self.assertEqual(ActionKind.DISMISS, action.kind)
        self.assertEqual((0.50, 0.93), action.point)

        later_unknown = VisionObservation(ScreenKind.UNKNOWN, 0.1)
        action = self.machine.update(later_unknown, 11.90)
        self.assertIsNotNone(action)
        self.assertEqual((0.50, 0.93), action.point, "按鈕特效期間變成 UNKNOWN 仍應維持收下按鈕座標")


if __name__ == "__main__":
    unittest.main()
