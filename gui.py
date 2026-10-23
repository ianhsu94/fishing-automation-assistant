from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
import ctypes
import tkinter as tk
from tkinter import messagebox, ttk

from config import load_config


BASE_DIR = Path(__file__).resolve().parent
_INSTANCE_MUTEX: int | None = None
_WINDOW_TITLE = "釣魚自動化助手"


def acquire_single_instance() -> bool:
    """Windows 單一實例鎖；重複啟動時喚回既有視窗。"""
    global _INSTANCE_MUTEX
    if os.name != "nt":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_bool
    handle = kernel32.CreateMutexW(None, False, "Local\\FishingAutomationAssistant.UI.v1")
    if not handle:
        return True
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
        user32.FindWindowW.restype = ctypes.c_void_p
        existing = user32.FindWindowW(None, _WINDOW_TITLE)
        if existing:
            user32.ShowWindow(existing, 9)  # SW_RESTORE
            user32.SetForegroundWindow(existing)
        return False
    _INSTANCE_MUTEX = int(handle)
    return True


def release_single_instance() -> None:
    global _INSTANCE_MUTEX
    if os.name == "nt" and _INSTANCE_MUTEX is not None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle(ctypes.c_void_p(_INSTANCE_MUTEX))
        _INSTANCE_MUTEX = None


def resolve_python() -> Path:
    virtual_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if virtual_python.is_file():
        return virtual_python
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        console_python = executable.with_name("python.exe")
        if console_python.is_file():
            return console_python
    return executable


class FishingBotApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(_WINDOW_TITLE)
        self.root.geometry("800x720")
        self.root.minsize(700, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.process: subprocess.Popen[str] | None = None
        self.process_mode = ""
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.status_var = tk.StringVar(value="待命")
        self.device_var = tk.StringVar(value="裝置：尚未檢查")
        self.mode_var = tk.StringVar(value="模式：未啟動")

        self._configure_styles()
        self._build_ui()
        self.root.after(100, self._drain_messages)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Microsoft JhengHei UI", 10), foreground="#4b5563")
        style.configure("Status.TLabel", font=("Microsoft JhengHei UI", 11, "bold"), foreground="#075985")
        style.configure("TButton", font=("Microsoft JhengHei UI", 10), padding=(13, 8))
        style.configure("Primary.TButton", font=("Microsoft JhengHei UI", 10, "bold"), foreground="white", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])
        style.configure("Danger.TButton", font=("Microsoft JhengHei UI", 10, "bold"), foreground="white", background="#dc2626")
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#fca5a5")])

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="釣魚自動化助手", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="先使用單次辨識與乾跑確認畫面，再啟用正式點擊。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        status_frame = ttk.LabelFrame(container, text="目前狀態", padding=12)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.device_var).grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Label(status_frame, textvariable=self.mode_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=14)
        controls.columnconfigure((0, 1, 2), weight=1)

        self.check_button = ttk.Button(controls, text="檢查 ADB 連線", command=self.check_connection)
        self.check_button.grid(row=0, column=0, padx=(0, 6), pady=5, sticky="ew")
        self.once_button = ttk.Button(controls, text="單次畫面辨識", command=self.run_once)
        self.once_button.grid(row=0, column=1, padx=6, pady=5, sticky="ew")
        self.debug_button = ttk.Button(controls, text="開啟除錯資料夾", command=self.open_debug_folder)
        self.debug_button.grid(row=0, column=2, padx=(6, 0), pady=5, sticky="ew")

        self.dry_button = ttk.Button(controls, text="開始乾跑", command=self.start_dry_run, style="Primary.TButton")
        self.dry_button.grid(row=1, column=0, padx=(0, 6), pady=5, sticky="ew")
        self.auto_button = ttk.Button(controls, text="開始自動點擊", command=self.start_clicking, style="Danger.TButton")
        self.auto_button.grid(row=1, column=1, padx=6, pady=5, sticky="ew")
        self.stop_button = ttk.Button(controls, text="停止", command=self.stop_process, state="disabled")
        self.stop_button.grid(row=1, column=2, padx=(6, 0), pady=5, sticky="ew")

        log_frame = ttk.LabelFrame(container, text="執行紀錄", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            height=16,
            font=("Consolas", 9),
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="white",
            relief="flat",
            padx=9,
            pady=9,
            state="disabled",
        )
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(
            container,
            text="安全提醒：正式模式會操作遊戲；可隨時按「停止」。每輪截圖會永久清除。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(10, 0))

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool, mode: str = "") -> None:
        normal_state = "disabled" if running else "normal"
        for button in (self.check_button, self.once_button, self.dry_button, self.auto_button):
            button.configure(state=normal_state)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.mode_var.set(f"模式：{mode}" if running else "模式：未啟動")

    def check_connection(self) -> None:
        if self.process is not None:
            return
        self.status_var.set("正在檢查 ADB…")
        self._append_log("[UI] 檢查設定檔中的 ADB 裝置")

        def worker() -> None:
            try:
                config = load_config(BASE_DIR / "config.json")
                executable = config.adb.executable
                resolved_executable = shutil.which(executable) or executable
                serial = config.adb.serial
                completed = subprocess.run(
                    [resolved_executable, "-s", serial, "get-state"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=config.adb.command_timeout_seconds,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode == 0 and completed.stdout.strip() == "device":
                    self.messages.put(("connection", (True, serial, "ADB 裝置已連線")))
                else:
                    detail = completed.stderr.strip() or completed.stdout.strip() or "無回應"
                    self.messages.put(("connection", (False, serial, detail)))
            except Exception as exc:  # UI 邊界：錯誤必須顯示給使用者
                self.messages.put(("connection", (False, "未知", str(exc))))

        threading.Thread(target=worker, daemon=True).start()

    def run_once(self) -> None:
        debug_dir = BASE_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        self._launch(["--once", "--debug-output", str(debug_dir / "current.png")], "單次辨識")

    def start_dry_run(self) -> None:
        self._launch([], "乾跑（不點擊）")

    def start_clicking(self) -> None:
        confirmed = messagebox.askyesno(
            "確認啟用自動點擊",
            "程式將透過 ADB 操作遊戲。\n\n請確認單次辨識與乾跑結果正確，並確認遊戲規範允許自動化。是否繼續？",
            parent=self.root,
        )
        if confirmed:
            self._launch(["--click"], "正式自動點擊")

    def _launch(self, arguments: list[str], mode: str) -> None:
        if self.process is not None:
            messagebox.showinfo("程式執行中", "請先停止目前工作。", parent=self.root)
            return
        command = [str(resolve_python()), "-u", str(BASE_DIR / "main.py"), *arguments]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.process = None
            messagebox.showerror("啟動失敗", str(exc), parent=self.root)
            return

        self.process_mode = mode
        self.status_var.set("執行中")
        self._set_running(True, mode)
        self._append_log(f"[UI] 啟動：{mode}")
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.messages.put(("log", line.rstrip()))
            return_code = process.wait()
            self.messages.put(("finished", return_code))
        except Exception as exc:
            self.messages.put(("log", f"[UI] 無法讀取子程序輸出：{exc}"))
            self.messages.put(("finished", -1))

    def stop_process(self) -> None:
        process = self.process
        if process is None:
            return
        self.status_var.set("正在停止…")
        self._append_log("[UI] 發出停止要求")
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        except OSError:
            pass
        self.root.after(1800, self._force_stop_if_needed)

    def _force_stop_if_needed(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            self.root.after(900, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()

    def open_debug_folder(self) -> None:
        debug_dir = BASE_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(debug_dir)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("無法開啟資料夾", str(exc), parent=self.root)

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    line = str(payload)
                    self._append_log(line)
                    if "狀態 " in line:
                        self.status_var.set(line.split("INFO", 1)[-1].strip())
                    elif "啟動失敗" in line or "停止以避免誤點" in line:
                        self.status_var.set("錯誤，已停止")
                elif kind == "connection":
                    ok, serial, detail = payload  # type: ignore[misc]
                    self.device_var.set(f"裝置：{serial}（{'已連線' if ok else '未連線'}）")
                    self.status_var.set("ADB 連線正常" if ok else "ADB 連線失敗")
                    self._append_log(f"[ADB] {detail}")
                elif kind == "finished":
                    return_code = int(payload)
                    self._append_log(f"[UI] 程序結束，代碼 {return_code}")
                    self.process = None
                    self.process_mode = ""
                    self.status_var.set("已停止" if return_code in (0, 1) else "執行失敗")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_messages)

    def on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("結束程式", "自動化仍在執行，是否停止並關閉？", parent=self.root):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        self.root.destroy()


def main() -> int:
    if not acquire_single_instance():
        return 0
    try:
        root = tk.Tk()
        FishingBotApp(root)
        root.mainloop()
    finally:
        release_single_instance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
