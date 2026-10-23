from __future__ import annotations

import os
import shutil
import subprocess
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from config import AdbConfig


class AdbError(RuntimeError):
    pass


class _H264ScreenStream:
    """以單一 screenrecord/H.264 串流持續取得最新畫面，避免每幀重啟 ADB。"""

    def __init__(self, client: "AdbClient") -> None:
        self.client = client
        self.config = client.config
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame: np.ndarray | None = None
        self.generation = 0
        self.last_delivered_generation = 0
        self.error: str | None = None
        self.server: socket.socket | None = None
        self.connection: socket.socket | None = None
        self.capture: cv2.VideoCapture | None = None
        self.adb_process: subprocess.Popen[bytes] | None = None
        self.producer_thread: threading.Thread | None = None
        self.reader_thread: threading.Thread | None = None
        self.pid_file = "/data/local/tmp/fishingbot-screenrecord.pid"

    def start(self) -> None:
        self.client._stop_remote_screenrecord(self.pid_file)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(self.config.command_timeout_seconds)
        self.server = server
        port = int(server.getsockname()[1])

        self.producer_thread = threading.Thread(target=self._produce, name="adb-h264-producer", daemon=True)
        self.producer_thread.start()

        # 要求 FFmpeg 不累積已過時的解碼影格；計時遊戲寧可丟舊幀，也要拿最新畫面。
        option_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        old_options = os.environ.get(option_name)
        os.environ[option_name] = "fflags;nobuffer|flags;low_delay"
        try:
            capture = cv2.VideoCapture(f"tcp://127.0.0.1:{port}", cv2.CAP_FFMPEG)
        finally:
            if old_options is None:
                os.environ.pop(option_name, None)
            else:
                os.environ[option_name] = old_options
        if not capture.isOpened():
            self.close()
            raise AdbError("無法開啟 ADB H.264 畫面串流")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        self.reader_thread = threading.Thread(target=self._read_frames, name="adb-h264-reader", daemon=True)
        self.reader_thread.start()

        deadline = time.monotonic() + self.config.command_timeout_seconds
        with self.condition:
            while self.latest_frame is None and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
        if self.latest_frame is None:
            detail = self.error or "等待第一張影像逾時"
            self.close()
            raise AdbError(f"ADB H.264 畫面串流啟動失敗：{detail}")

        # OpenCV 啟動時會暫存少量探測影格；讓讀取執行緒先排空，避免第一張是舊畫面。
        time.sleep(0.12)

    def _produce(self) -> None:
        try:
            assert self.server is not None
            connection, _ = self.server.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection = connection
            while not self.stop_event.is_set():
                command = (
                    f"echo $$ > {self.pid_file}; exec screenrecord "
                    f"--output-format=h264 --bit-rate {self.config.stream_bitrate} "
                    f"--size {self.config.stream_width}x{self.config.stream_height} "
                    "--time-limit 180 -"
                )
                creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                process = subprocess.Popen(
                    [
                        str(self.client.executable),
                        "-s",
                        self.config.serial,
                        "exec-out",
                        "sh",
                        "-c",
                        command,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                self.adb_process = process
                assert process.stdout is not None
                try:
                    while not self.stop_event.is_set():
                        chunk = process.stdout.read1(4096)
                        if not chunk:
                            break
                        connection.sendall(chunk)
                finally:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    self.adb_process = None
                if not self.stop_event.is_set():
                    time.sleep(0.05)
        except (OSError, TimeoutError) as exc:
            if not self.stop_event.is_set():
                self._set_error(f"串流傳輸中斷：{exc}")

    def _read_frames(self) -> None:
        assert self.capture is not None
        while not self.stop_event.is_set():
            try:
                ok, frame = self.capture.read()
            except cv2.error as exc:
                if not self.stop_event.is_set():
                    self._set_error(f"H.264 解碼錯誤：{exc}")
                return
            if not ok or frame is None or frame.size == 0:
                if not self.stop_event.is_set():
                    self._set_error("H.264 解碼停止")
                return
            with self.condition:
                self.latest_frame = frame
                self.generation += 1
                self.condition.notify_all()

    def _set_error(self, message: str) -> None:
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def screenshot(self) -> np.ndarray:
        deadline = time.monotonic() + self.config.command_timeout_seconds
        with self.condition:
            while self.generation <= self.last_delivered_generation and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdbError("等待下一張 H.264 畫面逾時")
                self.condition.wait(remaining)
            if self.error is not None:
                raise AdbError(self.error)
            if self.latest_frame is None:
                raise AdbError("H.264 串流尚無畫面")
            self.last_delivered_generation = self.generation
            return self.latest_frame.copy()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        process = self.adb_process
        if process is not None and process.poll() is None:
            process.terminate()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            self.connection = None
        if self.server is not None:
            self.server.close()
            self.server = None
        self.client._stop_remote_screenrecord(self.pid_file)
        for thread in (self.producer_thread, self.reader_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        if process is not None and process.poll() is None:
            process.kill()
        # VideoCapture.read() 與 release() 不可同時進行；等讀取執行緒退出後才釋放，
        # 避免停止程式時觸發 FFmpeg async_lock assertion。
        if self.capture is not None:
            if self.reader_thread is None or not self.reader_thread.is_alive():
                self.capture.release()
            self.capture = None


class AdbClient:
    def __init__(self, config: AdbConfig) -> None:
        self.config = config
        configured = Path(config.executable).expanduser()
        resolved = configured if configured.is_file() else shutil.which(config.executable)
        self.executable = Path(resolved) if resolved else configured
        if not self.executable.is_file():
            raise AdbError(f"找不到 ADB：{self.executable}")
        self._stream: _H264ScreenStream | None = None
        self._tap_process: subprocess.Popen[bytes] | None = None
        self._tap_lock = threading.Lock()
        self._device_size: tuple[int, int] | None = None

    def _run(self, args: list[str], *, binary: bool = False) -> subprocess.CompletedProcess[bytes]:
        command = [str(self.executable), "-s", self.config.serial, *args]
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.command_timeout_seconds,
                check=False,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB 指令逾時：{' '.join(command)}") from exc
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise AdbError(f"ADB 指令失敗 ({result.returncode})：{message}")
        return result

    def ensure_ready(self) -> None:
        result = self._run(["get-state"])
        state = result.stdout.decode("utf-8", errors="replace").strip()
        if state != "device":
            raise AdbError(f"裝置 {self.config.serial} 尚未就緒：{state or '無回應'}")
        # wm size 回報實體橫向尺寸，無法代表旋轉後的輸入座標；以真實截圖方向為準。
        try:
            probe = self._raw_screenshot()
        except (AdbError, ValueError, struct.error):
            probe = self._png_screenshot()
        self._device_size = (int(probe.shape[1]), int(probe.shape[0]))

    def screenshot(self) -> np.ndarray:
        if self.config.screenshot_mode in ("h264", "stream"):
            if self._stream is None:
                self._stream = _H264ScreenStream(self)
                self._stream.start()
            return self._stream.screenshot()
        if self.config.screenshot_mode == "raw":
            try:
                return self._raw_screenshot()
            except (AdbError, ValueError, struct.error):
                # 少數 Android 版本不支援預期的 raw 格式，安全退回 PNG。
                pass
        return self._png_screenshot()

    def _raw_screenshot(self) -> np.ndarray:
        result = self._run(["exec-out", "screencap"], binary=True)
        data = result.stdout
        if len(data) < 16:
            raise AdbError("ADB raw 截圖資料過短")
        width, height, pixel_format = struct.unpack("<3I", data[:12])
        if not (1 <= width <= 10000 and 1 <= height <= 10000):
            raise AdbError(f"ADB raw 截圖尺寸異常：{width}x{height}")
        pixel_bytes = width * height * 4
        header_size = 16 if len(data) == pixel_bytes + 16 else 12
        if len(data) != pixel_bytes + header_size:
            raise AdbError(
                f"ADB raw 截圖長度不符：格式={pixel_format}，收到={len(data)}，預期={pixel_bytes + header_size}"
            )
        rgba = np.frombuffer(data, dtype=np.uint8, offset=header_size).reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _png_screenshot(self) -> np.ndarray:
        result = self._run(["exec-out", "screencap", "-p"], binary=True)
        if not result.stdout:
            raise AdbError("ADB 截圖是空資料")
        encoded = np.frombuffer(result.stdout, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise AdbError("OpenCV 無法解碼 ADB 截圖")
        return image

    def tap_pixels(self, x: int, y: int) -> None:
        command = f"input tap {int(x)} {int(y)}\n".encode("ascii")
        with self._tap_lock:
            for attempt in range(2):
                process = self._tap_process
                if process is None or process.poll() is not None or process.stdin is None:
                    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    process = subprocess.Popen(
                        [str(self.executable), "-s", self.config.serial, "shell"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=creationflags,
                        bufsize=0,
                    )
                    self._tap_process = process
                try:
                    assert process.stdin is not None
                    process.stdin.write(command)
                    process.stdin.flush()
                    return
                except (BrokenPipeError, OSError):
                    self._tap_process = None
                    if attempt:
                        raise AdbError("ADB 低延遲點擊通道中斷")

    def tap_normalized(self, point: tuple[float, float], frame_shape: tuple[int, ...]) -> tuple[int, int]:
        if self._device_size is None:
            height, width = frame_shape[:2]
        else:
            width, height = self._device_size
        x = min(width - 1, max(0, round(point[0] * width)))
        y = min(height - 1, max(0, round(point[1] * height)))
        self.tap_pixels(x, y)
        return x, y

    def _stop_remote_screenrecord(self, pid_file: str) -> None:
        try:
            result = self._run(["shell", "cat", pid_file])
            pid = result.stdout.decode("ascii", errors="ignore").strip()
            if pid.isdigit():
                self._run(["shell", "kill", pid])
        except AdbError:
            pass
        try:
            self._run(["shell", "rm", "-f", pid_file])
        except AdbError:
            pass

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        with self._tap_lock:
            process = self._tap_process
            self._tap_process = None
            if process is not None and process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write(b"exit\n")
                        process.stdin.flush()
                    process.wait(timeout=1.0)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
