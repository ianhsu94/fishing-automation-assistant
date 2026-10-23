from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from adb_client import AdbClient, AdbError
from config import AppConfig, load_config
from image_io import read_image, write_image
from models import ActionKind, BotAction
from state_machine import FishingStateMachine
from vision import VisionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LDPlayer 釣魚大賽自動化助手")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")), help="JSON 設定檔")
    parser.add_argument("--click", action="store_true", help="啟用 ADB 點擊；未指定時一律乾跑")
    parser.add_argument("--once", action="store_true", help="只擷取與辨識一次")
    parser.add_argument("--input", help="改讀本機圖片，不連接 ADB")
    parser.add_argument("--debug-output", help="儲存本次標記圖的位置")
    return parser.parse_args()


def configure_logging(config: AppConfig) -> logging.Logger:
    config.output.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fishing_bot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        config.output.log_dir / "bot.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def save_debug(config: AppConfig, detector: VisionDetector, frame, observation, prefix: str) -> Path:
    config.output.debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = config.output.debug_dir / f"{stamp}-{prefix}-{observation.kind.value}.png"
    if not write_image(path, detector.annotate(frame, observation)):
        raise OSError(f"無法儲存除錯圖：{path}")
    return path


def purge_debug_images(directory: Path, paths: list[Path] | None = None) -> int:
    """永久刪除執行期截圖；範圍嚴格限制在設定的 debug 資料夾。"""
    directory = directory.resolve()
    candidates = paths if paths is not None else list(directory.glob("*")) if directory.exists() else []
    removed = 0
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.parent != directory or resolved.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            resolved.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def execute_action(
    adb: AdbClient,
    frame,
    action: BotAction,
    enabled: bool,
    logger: logging.Logger,
    scheduled_from: float | None = None,
) -> None:
    if not enabled:
        logger.info(
            "DRY-RUN 動作=%s 座標=%s 延遲=%.3f秒 原因=%s",
            action.kind.value,
            action.point,
            action.delay_seconds,
            action.reason,
        )
        return
    wait_seconds = action.delay_seconds
    if scheduled_from is not None:
        wait_seconds = max(0.0, wait_seconds - (time.monotonic() - scheduled_from))
    if wait_seconds > 0.0:
        logger.info("已排定 %.3f 秒後送出計時點擊", wait_seconds)
        deadline = time.perf_counter() + wait_seconds
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            # 先低耗能等待，最後 2ms 再讓出時間片微調，減少 Windows sleep 超時。
            time.sleep(max(0.0, remaining - 0.002) if remaining > 0.004 else 0.0)
    x, y = adb.tap_normalized(action.point, frame.shape)
    logger.info("CLICK 動作=%s 像素=(%d,%d) 原因=%s", action.kind.value, x, y, action.reason)


def analyze_one(args: argparse.Namespace, config: AppConfig, detector: VisionDetector, logger: logging.Logger) -> int:
    if args.input:
        frame = read_image(Path(args.input).expanduser().resolve(), cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("無法讀取圖片：%s", args.input)
            return 2
    else:
        adb = AdbClient(config.adb)
        try:
            adb.ensure_ready()
            frame = adb.screenshot()
        finally:
            adb.close()
    observation = detector.analyze(frame)
    print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))
    output = Path(args.debug_output).expanduser().resolve() if args.debug_output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not write_image(output, detector.annotate(frame, observation)):
            logger.error("無法儲存標記圖：%s", output)
            return 2
        logger.info("已儲存標記圖：%s", output)
    return 0


def run_loop(args: argparse.Namespace, config: AppConfig, logger: logging.Logger) -> int:
    adb = AdbClient(config.adb)
    try:
        adb.ensure_ready()
        return _run_loop(args, config, logger, adb)
    finally:
        adb.close()
        if config.output.cleanup_debug_each_round:
            purge_debug_images(config.output.debug_dir)


def _run_loop(args: argparse.Namespace, config: AppConfig, logger: logging.Logger, adb: AdbClient) -> int:
    detector = VisionDetector(config.vision)
    machine = FishingStateMachine(config.control, config.vision.rois["timing_bar"])
    stop_requested = False

    def stop_handler(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop_handler)

    period = 1.0 / max(0.2, config.control.target_fps)
    capture_errors = 0
    previous_state = machine.state
    fps_window_started = time.monotonic()
    fps_frames = 0
    capture_seconds = 0.0
    round_debug_paths: list[Path] = []
    round_started = False
    last_timing_diagnostic_at = -1e9
    if config.output.cleanup_debug_each_round:
        removed = purge_debug_images(config.output.debug_dir)
        if removed:
            logger.info("啟動時永久刪除舊除錯圖 %d 張", removed)
    logger.info(
        "模式=%s 裝置=%s 截圖=%s 目標FPS=%.2f；Ctrl+C 可安全停止",
        "CLICK" if args.click else "DRY-RUN",
        config.adb.serial,
        config.adb.screenshot_mode,
        config.control.target_fps,
    )

    while not stop_requested:
        started = time.monotonic()
        try:
            capture_started = time.monotonic()
            frame = adb.screenshot()
            capture_seconds += time.monotonic() - capture_started
            fps_frames += 1
            capture_errors = 0
        except AdbError as exc:
            capture_errors += 1
            logger.error("截圖失敗 %d/%d：%s", capture_errors, config.control.max_consecutive_capture_errors, exc)
            if capture_errors >= config.control.max_consecutive_capture_errors:
                logger.error("連續截圖失敗，停止以避免誤點")
                return 3
            time.sleep(min(1.0, period))
            continue

        observation = detector.analyze(frame)
        now = time.monotonic()
        action = machine.update(observation, now)
        state_changed = machine.state != previous_state
        if state_changed:
            logger.info("狀態 %s -> %s；畫面=%s 信心=%.3f", previous_state.value, machine.state.value, observation.kind.value, observation.confidence)
            previous_state = machine.state

        # 魚一釣上、進入結算狀態的同一刻就永久刪除本輪所有執行期截圖。
        # 直接清空 debug 內的圖片，也涵蓋 UI 單次辨識留下的 current.png；
        # 不等結算按鈕點完，且下方不會再為 RESULT 另存新圖。
        if state_changed and machine.state.value == "RESULT" and config.output.cleanup_debug_each_round:
            removed = purge_debug_images(config.output.debug_dir)
            round_debug_paths.clear()
            round_started = False
            logger.info("魚已釣上，本輪截圖已永久刪除 %d 張", removed)

        if action is not None:
            if action.kind == ActionKind.START and round_started and config.output.cleanup_debug_each_round:
                removed = purge_debug_images(config.output.debug_dir)
                round_debug_paths.clear()
                if removed:
                    logger.info("上一輪結束，永久刪除截圖 %d 張", removed)
            execute_action(adb, frame, action, args.click, logger, scheduled_from=now)
            if action.kind == ActionKind.START:
                round_started = True
            elif action.kind == ActionKind.DISMISS:
                round_started = False
                if config.output.cleanup_debug_each_round:
                    removed = purge_debug_images(config.output.debug_dir)
                    round_debug_paths.clear()
                    if removed:
                        logger.info("本輪結束，永久刪除截圖 %d 張", removed)
        elif machine.state.value == "TIMING" and now - last_timing_diagnostic_at >= 1.0:
            logger.info("計時等待 %s；畫面=%s", machine.last_timing_decision, observation.kind.value)
            last_timing_diagnostic_at = now

        should_save = (
            state_changed and config.output.save_debug_on_state_change
        ) or (
            action is not None and config.output.save_debug_on_action
        )
        if machine.state.value == "RESULT":
            should_save = False
        if should_save:
            path = save_debug(config, detector, frame, observation, machine.state.value.lower())
            round_debug_paths.append(path)
            logger.info("除錯圖：%s", path)

        if now - fps_window_started >= 3.0:
            window_seconds = now - fps_window_started
            logger.info(
                "效能 實測FPS=%.2f 平均截圖=%.0fms",
                fps_frames / window_seconds,
                capture_seconds * 1000.0 / max(1, fps_frames),
            )
            fps_window_started = now
            fps_frames = 0
            capture_seconds = 0.0

        elapsed = time.monotonic() - started
        if elapsed < period:
            time.sleep(period - elapsed)

    logger.info("收到停止要求，已停止；不再送出點擊")
    if config.output.cleanup_debug_each_round:
        removed = purge_debug_images(config.output.debug_dir)
        if removed:
            logger.info("停止時永久刪除剩餘截圖 %d 張", removed)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config = load_config(args.config)
    logger = configure_logging(config)
    detector = VisionDetector(config.vision)
    try:
        if args.once or args.input:
            return analyze_one(args, config, detector, logger)
        return run_loop(args, config, logger)
    except (AdbError, ValueError, OSError) as exc:
        logger.error("啟動失敗：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
