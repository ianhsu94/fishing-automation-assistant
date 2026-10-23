from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import Roi, VisionConfig
from image_io import read_image
from models import BarObservation, ScreenKind, VisionObservation


class VisionDetector:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.templates = self._load_templates(config.template_dir)

    @staticmethod
    def _load_templates(directory: Path) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for name in ("title", "result_banner", "bobber"):
            path = directory / f"{name}.png"
            if path.is_file():
                image = read_image(path, cv2.IMREAD_COLOR)
                if image is not None:
                    templates[name] = image
        return templates

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        size = (self.config.canonical_width, self.config.canonical_height)
        interpolation = cv2.INTER_AREA if frame.shape[1] > size[0] else cv2.INTER_LINEAR
        return cv2.resize(frame, size, interpolation=interpolation)

    @staticmethod
    def _crop(frame: np.ndarray, roi: Roi) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = roi.pixels(width, height)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    @staticmethod
    def _fraction(hsv: np.ndarray, ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> float:
        if hsv.size == 0:
            return 0.0
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
        return float(cv2.countNonZero(mask)) / float(mask.size)

    @staticmethod
    def _edge_template_score(search: np.ndarray, template: np.ndarray, scales: tuple[float, ...]) -> tuple[float, tuple[int, int] | None, tuple[int, int] | None]:
        if search.size == 0 or template.size == 0:
            return 0.0, None, None
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        search_edge = cv2.Canny(search_gray, 70, 170)
        best_score = -1.0
        best_at: tuple[int, int] | None = None
        best_size: tuple[int, int] | None = None
        for scale in scales:
            width = max(8, round(template.shape[1] * scale))
            height = max(8, round(template.shape[0] * scale))
            if width > search.shape[1] or height > search.shape[0]:
                continue
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            template_edge = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 70, 170)
            result = cv2.matchTemplate(search_edge, template_edge, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = float(score)
                best_at = location
                best_size = (width, height)
        return max(0.0, best_score), best_at, best_size

    def _template_in_roi(self, frame: np.ndarray, template_name: str, roi_name: str) -> float:
        template = self.templates.get(template_name)
        if template is None:
            return 0.0
        crop, _ = self._crop(frame, self.config.rois[roi_name])
        score, _, _ = self._edge_template_score(crop, template, (0.82, 0.90, 1.0, 1.10, 1.20))
        return score

    def detect_bar(self, frame: np.ndarray) -> BarObservation:
        crop, (x1, y1, x2, y2) = self._crop(frame, self.config.rois["timing_bar"])
        if crop.size == 0:
            return BarObservation()

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 真正的計時橫條一定有大片深藍軌道。只靠黃區與紅色浮標會把金幣動畫、
        # 魚竿或圓形按鈕內的圖案誤認成計時條。
        track_mask = cv2.inRange(hsv, np.array((85, 50, 20)), np.array((145, 255, 180)))
        track_fraction = np.count_nonzero(track_mask) / max(1, track_mask.size)
        if track_fraction < self.config.min_bar_track_fraction:
            return BarObservation()
        track_joined = cv2.morphologyEx(
            track_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, round(crop.shape[1] * 0.08)), 5)),
        )
        track_contours, _ = cv2.findContours(track_joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        track_candidate: tuple[int, int, int, int] | None = None
        for contour in track_contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            if tw >= crop.shape[1] * 0.35 and tw / max(1, th) >= 2.0:
                if track_candidate is None or tw * th > track_candidate[2] * track_candidate[3]:
                    track_candidate = (tx, ty, tw, th)
        if track_candidate is None:
            return BarObservation()
        tx, _, tw, _ = track_candidate
        track_left = (x1 + tx) / frame.shape[1]
        track_right = (x1 + tx + tw) / frame.shape[1]

        yellow_mask = cv2.inRange(hsv, np.array((17, 95, 120)), np.array((42, 255, 255)))
        # 紅色浮標會蓋住黃色長條，把同一個目標切成左右兩個輪廓。使用只在水平方向
        # 較寬的 closing 把遮擋缺口接回去，否則程式會誤以為浮標永遠位於黃區外。
        bridge_width = max(11, round(crop.shape[1] * 0.12))
        if bridge_width % 2 == 0:
            bridge_width += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_width, 3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate: tuple[float, tuple[int, int, int, int]] | None = None
        roi_area = float(crop.shape[0] * crop.shape[1])
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            aspect = bw / max(1.0, float(bh))
            if bw < crop.shape[1] * self.config.min_yellow_width_fraction or bh < 4 or aspect < 1.25:
                continue
            fill = area / max(1.0, float(bw * bh))
            score = min(1.0, area / max(1.0, roi_area * 0.035)) * min(1.0, fill / 0.45)
            if candidate is None or score > candidate[0]:
                candidate = (score, (bx, by, bw, bh))

        target_left: float | None = None
        target_right: float | None = None
        yellow_score = 0.0
        if candidate is not None:
            yellow_score, (bx, _, bw, _) = candidate
            target_left = (x1 + bx) / frame.shape[1]
            target_right = (x1 + bx + bw) / frame.shape[1]

        bobber_score = 0.0
        bobber_x: float | None = None
        bobber_half_width: float | None = None

        # 浮標的紅色直條比背景穩定；先以色彩取得位置，再以模板分數補強信心。
        red_a = cv2.inRange(hsv, np.array((0, 105, 80)), np.array((12, 255, 255)))
        red_b = cv2.inRange(hsv, np.array((168, 105, 80)), np.array((179, 255, 255)))
        red = cv2.bitwise_or(red_a, red_b)
        column_strength = np.count_nonzero(red, axis=0)
        red_score = 0.0
        if column_strength.size:
            peak = int(column_strength.max())
            minimum_peak = max(5, round(crop.shape[0] * 0.10))
            if peak >= minimum_peak:
                peak_column = int(np.argmax(column_strength))
                threshold = max(minimum_peak, round(peak * 0.45))
                left = peak_column
                right = peak_column
                while left > 0 and column_strength[left - 1] >= threshold:
                    left -= 1
                while right + 1 < column_strength.size and column_strength[right + 1] >= threshold:
                    right += 1
                indices = np.arange(left, right + 1)
                weights = column_strength[left : right + 1].astype(np.float64)
                color_column = float(np.average(indices, weights=weights))
                bobber_x = (x1 + color_column) / frame.shape[1]
                bobber_half_width = (right - left + 1) / (2.0 * frame.shape[1])
                red_score = min(1.0, peak / max(1.0, crop.shape[0] * 0.24))

        bobber_score = red_score
        template = self.templates.get("bobber")
        if template is not None and red_score < self.config.min_bobber_score:
            template_score, location, size = self._edge_template_score(
                crop, template, (0.78, 0.88, 1.0, 1.12, 1.24)
            )
            bobber_score = max(red_score, template_score)
            if bobber_x is None and location is not None and size is not None:
                bobber_x = (x1 + location[0] + size[0] / 2.0) / frame.shape[1]
        elif template is None:
            bobber_score = red_score

        confidence = min(1.0, 0.48 * yellow_score + 0.52 * bobber_score)
        return BarObservation(
            target_left=target_left,
            target_right=target_right,
            bobber_x=bobber_x,
            bobber_score=bobber_score,
            confidence=confidence,
            track_left=track_left,
            track_right=track_right,
            bobber_half_width=bobber_half_width,
        )

    def analyze(self, input_frame: np.ndarray) -> VisionObservation:
        frame = self.normalize(input_frame)
        action, _ = self._crop(frame, self.config.rois["action"])
        prompt, _ = self._crop(frame, self.config.rois["prompt"])
        result_roi, _ = self._crop(frame, self.config.rois["result"])

        action_hsv = cv2.cvtColor(action, cv2.COLOR_BGR2HSV)
        prompt_hsv = cv2.cvtColor(prompt, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result_roi, cv2.COLOR_BGR2HSV)

        green = self._fraction(action_hsv, [((38, 85, 80), (90, 255, 255))])
        purple = self._fraction(action_hsv, [((125, 70, 70), (172, 255, 255))])
        prompt_purple = self._fraction(prompt_hsv, [((125, 70, 60), (172, 255, 255))])
        prompt_yellow = self._fraction(prompt_hsv, [((17, 95, 120), (42, 255, 255))])
        timing_success_white = self._fraction(prompt_hsv, [((0, 0, 210), (179, 70, 255))])
        result_dark = self._fraction(result_hsv, [((0, 0, 0), (179, 255, 78))])
        result_blue = self._fraction(result_hsv, [((88, 70, 70), (132, 255, 255))])
        result_purple = self._fraction(result_hsv, [((125, 70, 70), (172, 255, 255))])
        result_orange = self._fraction(result_hsv, [((4, 95, 95), (38, 255, 255))])

        title_score = 0.0
        result_template_score = 0.0
        bar = BarObservation()

        features = {
            "title_score": title_score,
            "result_score": result_template_score,
            "ready_green": green,
            "action_purple": purple,
            "prompt_purple": prompt_purple,
            "prompt_yellow": prompt_yellow,
            "timing_success_white": timing_success_white,
            "result_dark": result_dark,
            "result_blue": result_blue,
            "result_purple": result_purple,
            "result_orange": result_orange,
            "bar_confidence": 0.0,
            "active_game_scene": 0.0,
        }

        # 「開心收下」仍保留深色結算底板，但底部會多出大型綠色按鈕。
        result_confirm_heuristic = (
            green >= self.config.min_ready_green_fraction
            and result_dark > 0.55
            and (result_blue > 0.05 or result_purple > 0.12)
        )
        if result_confirm_heuristic:
            confidence = min(1.0, 0.45 + green + result_dark * 0.25)
            return VisionObservation(ScreenKind.RESULT_CONFIRM, confidence, bar, features)

        result_heuristic = result_dark > 0.34 and result_blue > 0.17 and green < 0.08 and purple < 0.08
        # 「刷新個人紀錄」等結算動畫可能完全沒有按鈕，底板也比一般結算頁亮，
        # 但仍有明顯的深色覆蓋層、大片藍色水面且沒有綠／紫操作按鈕。
        no_button_result_heuristic = (
            result_dark > 0.18
            and result_blue > 0.35
            and green < 0.08
            and purple < 0.08
            and prompt_purple < self.config.min_prompt_purple_fraction * 0.75
        )
        record_result_heuristic = (
            max(result_purple, result_orange) > 0.24
            and result_dark > 0.035
            and purple < self.config.min_action_purple_fraction * 1.2
        )
        if result_dark > 0.20 and result_blue > 0.10:
            result_template_score = self._template_in_roi(frame, "result_banner", "result")
            features["result_score"] = result_template_score
        if (
            result_template_score >= self.config.min_result_score
            or result_heuristic
            or no_button_result_heuristic
            or record_result_heuristic
        ):
            confidence = max(
                result_template_score,
                min(0.95, result_dark + max(result_blue * 0.5, result_purple, result_orange)),
            )
            return VisionObservation(ScreenKind.RESULT, confidence, bar, features)

        # 釣魚進行畫面中央長時間維持亮藍水面。先確認場景身分，避免 Android
        # 首頁、其他遊戲或廣告剛好含有紫色／黃色／紅色元素時被誤判並點擊。
        active_game_scene = (
            result_blue >= self.config.min_active_game_blue_fraction
            and result_dark <= self.config.max_active_game_dark_fraction
        )
        features["active_game_scene"] = 1.0 if active_game_scene else 0.0
        if not active_game_scene:
            title_score = self._template_in_roi(frame, "title", "title")
            features["title_score"] = title_score
            return VisionObservation(ScreenKind.UNKNOWN, title_score, bar, features)

        bite_ready = (
            purple >= self.config.min_action_purple_fraction
            and prompt_purple >= self.config.min_prompt_purple_fraction
            and prompt_yellow >= self.config.min_prompt_yellow_fraction
        )
        if bite_ready:
            confidence = min(1.0, purple + prompt_purple + prompt_yellow)
            return VisionObservation(ScreenKind.BITE, confidence, bar, features)

        # 計時成功的綠色閃光曾被 READY 規則誤判。只要仍有計時色彩，先檢查橫條。
        likely_timing = (
            purple >= self.config.min_action_purple_fraction * 0.35
            or prompt_yellow >= self.config.min_prompt_yellow_fraction * 0.50
        )
        if likely_timing:
            bar = self.detect_bar(frame)
            features["bar_confidence"] = bar.confidence

        timing_ready = (
            bar.valid
            and bar.bobber_score >= self.config.min_bobber_score
            and bar.target_left is not None
            and bar.target_right is not None
            and bar.target_right - bar.target_left >= self.config.min_yellow_width_fraction
        )
        if timing_ready:
            confidence = min(1.0, 0.55 * bar.bobber_score + 0.30 * bar.confidence + 0.15 * purple)
            return VisionObservation(ScreenKind.TIMING, confidence, bar, features)

        # READY 必須是乾淨的綠色主按鈕；計時中的紫色底板或黃條會排除此規則。
        clean_ready = (
            green >= self.config.min_ready_green_fraction
            and purple < self.config.min_action_purple_fraction * 0.75
            and prompt_yellow < self.config.min_prompt_yellow_fraction * 0.50
        )
        if clean_ready:
            confidence = min(1.0, green * 2.8)
            return VisionObservation(ScreenKind.READY, confidence, bar, features)

        if not likely_timing:
            bar = self.detect_bar(frame)
            features["bar_confidence"] = bar.confidence
            timing_ready = (
                bar.valid
                and bar.bobber_score >= self.config.min_bobber_score
                and bar.target_left is not None
                and bar.target_right is not None
                and bar.target_right - bar.target_left >= self.config.min_yellow_width_fraction
            )
            if timing_ready:
                confidence = min(1.0, 0.55 * bar.bobber_score + 0.30 * bar.confidence + 0.15 * purple)
                return VisionObservation(ScreenKind.TIMING, confidence, bar, features)

        title_score = self._template_in_roi(frame, "title", "title")
        features["title_score"] = title_score
        return VisionObservation(ScreenKind.UNKNOWN, max(title_score, bar.confidence), bar, features)

    def annotate(self, input_frame: np.ndarray, observation: VisionObservation) -> np.ndarray:
        frame = self.normalize(input_frame).copy()
        colors = {
            "title": (255, 180, 0),
            "action": (0, 220, 0),
            "prompt": (220, 0, 220),
            "timing_bar": (0, 220, 255),
            "result": (255, 80, 80),
        }
        for name, roi in self.config.rois.items():
            x1, y1, x2, y2 = roi.pixels(frame.shape[1], frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(name, (255, 255, 255)), 2)
            cv2.putText(frame, name, (x1 + 3, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(name, (255, 255, 255)), 1, cv2.LINE_AA)

        if observation.bar.target_left is not None and observation.bar.target_right is not None:
            x_left = round(observation.bar.target_left * frame.shape[1])
            x_right = round(observation.bar.target_right * frame.shape[1])
            y = round(self.config.rois["timing_bar"].y2 * frame.shape[0])
            cv2.line(frame, (x_left, y), (x_right, y), (0, 255, 255), 5)
        if observation.bar.bobber_x is not None:
            x = round(observation.bar.bobber_x * frame.shape[1])
            y1 = round(self.config.rois["timing_bar"].y1 * frame.shape[0])
            y2 = round(self.config.rois["timing_bar"].y2 * frame.shape[0])
            cv2.line(frame, (x, y1), (x, y2), (0, 0, 255), 3)

        label = f"{observation.kind.value}  conf={observation.confidence:.2f}"
        cv2.rectangle(frame, (8, 8), (390, 48), (0, 0, 0), -1)
        cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
