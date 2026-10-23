# LDPlayer 釣魚自動化助手

以 ADB 與 OpenCV 實作釣魚畫面辨識與自動點擊。

## 使用前必填的 ADB 設定

ADB 是 Android Debug Bridge，不是帳號或密碼；但實體裝置序號與本機安裝路徑可能透露裝置資訊，所以個人的 `config.json` 不會上傳到 GitHub。

1. 將 `config.example.json` 複製並命名為 `config.json`。
2. 開啟 `config.json`，修改最前面的兩個欄位：
   - 第 3 行 `adb.executable`：填入自己的 ADB 執行檔路徑，例如 `C:\LDPlayer\LDPlayer9\adb.exe`；若 ADB 已加入 PATH，可保留 `adb.exe`。
   - 第 4 行 `adb.serial`：填入自己的模擬器或 Android 裝置序號，例如 `emulator-5554`；可使用 `adb devices` 查詢。
   - 範例值會用 `【請填入……】` 括號醒目標示；請連同括號內的提示文字一起替換成自己的設定。
3. 雙擊 `啟動釣魚助手.bat`，先按「檢查 ADB 連線」。
4. 先執行單次辨識與乾跑，確認正確後才啟用正式點擊。

## 隱私保護

請勿提交自己的 `config.json`、`.env`、`logs/`、`debug/`、遊戲截圖或 `.venv/`。實際畫面可能包含玩家名稱、排行榜數字與帳號進度，因此 `samples/` 的圖片預設不會提交；公開前請再使用 `git status` 檢查。

完整安裝、校正與測試說明會隨程式碼一同加入。
