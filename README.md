# LDPlayer 釣魚自動化助手

這個專案依據提供的規格與 8 張遊戲畫面實作。它以 ADB 擷取 LDPlayer 畫面，用 OpenCV 判斷「可開始、上鉤提示、浮標判定、結算」狀態，再由有限狀態機控制點擊時機。

程式預設為乾跑模式：會辨識、記錄與輸出除錯圖，但不會點擊。只有明確加入 `--click` 才會送出 ADB 點擊。

## 使用者先填的本機設定（必讀）

`ADB` 是 Android Debug Bridge，不是帳號或密碼。不過 ADB 的實體裝置序號與電腦上的安裝路徑可能透露本機資訊，因此本專案不會把使用者自己的 `config.json` 上傳到 GitHub。

1. 將 `config.example.json` 複製一份並命名為 `config.json`。
2. 開啟 `config.json`，修改最前面的兩個欄位：
   - 第 3 行 `adb.executable`：填入自己的 ADB 執行檔路徑，例如 `C:\\LDPlayer\\LDPlayer9\\adb.exe`。若 `adb.exe` 已加入系統 PATH，可保留 `adb.exe`。
   - 第 4 行 `adb.serial`：填入自己的模擬器或 Android 裝置序號，例如 LDPlayer 常見的 `emulator-5554`。可執行 `adb devices` 查詢。
   - 範例值會用 `【請填入……】` 括號醒目標示；請連同括號內的提示文字一起替換成自己的設定。
3. 雙擊 `啟動釣魚助手.bat`，先按「檢查 ADB 連線」。
4. 先執行「單次畫面辨識」與「乾跑」，確認辨識正常後再開啟正式點擊。

請勿把自己的 `config.json`、`.env`、`logs/`、`debug/`、裝置截圖或 `.venv/` 提交到 Git。專案的 `.gitignore` 會排除這些本機資料；公開前仍應使用 `git status` 再檢查一次。

`samples/` 內的實際遊戲截圖也可能含有玩家名稱、排行榜數字與帳號進度，因此預設不會提交。公開儲存庫只保留 `samples/README.md`；沒有私人樣本時，依賴截圖的測試會自動略過，其餘單元測試仍會執行。

## 已實作

- 驗證使用者設定的 Android 裝置與 ADB 狀態
- 持續擷取畫面，截圖失敗三次即停止
- 以比例 ROI 與模板辨識五種畫面狀態
- HSV 黃色區段偵測與浮標模板比對
- 以速度與延遲估計浮標的預測位置
- READY、WAIT_BITE、TIMING、RESULT、RECOVERY 狀態機
- 穩定幀確認、點擊冷卻、逾時與重複點擊防護
- 乾跑、單張診斷、標記圖、檔案日誌與 Ctrl+C 安全停止
- 參考圖自動測試
- 使用持續 H.264 串流取代逐張 ADB 截圖，並以常駐 ADB shell 降低點擊延遲
- 辨識窄黃色區間，並排除計時成功綠光造成的 READY 誤判
- 辨識一般結算、刷新個人紀錄及其他結算後提示
- 每輪與停止時永久刪除執行期截圖；日誌採輪替上限避免磁碟持續成長

## 安裝

在此資料夾開啟 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

若目前的 Python 3.12 已安裝 OpenCV 與 NumPy，也可直接使用既有環境。

## 圖形介面與啟動檔

直接雙擊 `啟動釣魚助手.bat` 即可開啟簡易控制介面。介面提供：

- 檢查 ADB 連線
- 單次畫面辨識並產生標記圖
- 開始乾跑（不點擊）
- 開始正式自動點擊
- 停止目前工作
- 開啟除錯資料夾

正式自動點擊前會再次顯示確認視窗。關閉 UI 時若程式仍在執行，也會先詢問是否停止。

## 先做一次診斷

讓 LDPlayer 顯示釣魚小遊戲，再執行：

```powershell
python main.py --once --debug-output debug\current.png
```

終端會輸出 `ready`、`bite`、`timing`、`result` 或 `unknown`，並在 `debug\current.png` 畫出 ROI、黃色安全區與浮標位置。

目前擷取到的模擬器主畫面是 900 × 1600，並不在釣魚小遊戲內，因此正確結果應為 `unknown`。進入小遊戲後仍應先跑上面的單次診斷，確認比例 ROI 是否對齊。

## 乾跑與正式點擊

乾跑會持續辨識但不點擊：

```powershell
python main.py
```

觀察 `logs\bot.log` 與 `debug\`，確認四種畫面都能穩定辨識後，才啟用點擊：

```powershell
python main.py --click
```

任何時候按 `Ctrl+C` 都會停止主循環。連線中斷或連續截圖錯誤時，程式會停止點擊並退出。

## 結算畫面

程式確認進入結算階段後，只要尚未回到正常釣魚介面，就會每隔 0.5 秒點擊原本圓形按鈕位置 `[0.50, 0.80]`。這不依賴「開心收下」文字或特定獎勵版型；為避免誤點其他應用程式，通用點擊只會在已確認的結算階段生效。

## 校正設定

所有 ROI 與點擊位置都在 `config.json`，格式是相對於整張截圖的 `[x1, y1, x2, y2]` 或 `[x, y]`，範圍為 0 到 1。最常需要調整：

- `vision.rois.action`：底部圓形按鈕
- `vision.rois.prompt`：紫色上鉤提示
- `vision.rois.timing_bar`：水平浮標判定條
- `vision.rois.result`：獲魚結算區
- `control.taps`：開始、拉魚、判定及結算的點擊位置
- `control.latency_compensation_seconds`：H.264 畫面相對於遊戲真實狀態的延遲補償
- `control.tap_command_lead_seconds`：ADB 點擊指令進入遊戲前的提前量
- `control.max_timing_schedule_seconds`：允許提前排程計時點擊的最長等待時間
- `control.no_button_result_seconds`：計時後無按鈕、無計時條多久視為結算
- `control.min_valid_bobber_speed_per_second` / `max_bobber_speed_per_second`：排除反彈與換區造成的假速率範圍

目前預設使用低緩衝 `h264` 串流、30 FPS 目標循環及實測 0.34 秒畫面延遲補償。計時判斷會從連續畫面量出固定移動速率與左右方向，再依深藍軌道的真實邊界計算反彈，持續重算下一次進入黃色安全區的時間，直到剩餘 0.30 秒內才鎖定精準點擊，避免長距離預測誤差累積。即使畫面延遲期間已整段跨過最窄黃區，也不會在區外補點，而會計算反彈後下一次回程的進入時刻。辨識同時要求真正的深藍橫條軌道，避免把金幣動畫或按鈕圖案誤認成計時畫面。每次通過黃區只送出一次點擊；看到遊戲的綠色閃光或藍白 `Cool/Perfect` 命中回饋後，仍需黃區連續兩幀穩定才允許下一次點擊。若第一擊沒有命中回饋，程式會等待至少 0.75 秒，並確認整個浮標連續 3 幀完全離開黃區，才允許在下一次往返重試，不會在原地連點。計時點擊後若畫面持續 0.45 秒沒有按鈕或計時條，會視為結算並點擊原本圓形按鈕位置；「刷新個人紀錄」這類無按鈕結算頁也有獨立影像判斷。若 H.264 在特殊環境無法啟動，可將 `adb.screenshot_mode` 暫時改回 `raw`。程式會每三秒在紀錄中顯示實測值。

有綠色「開心收下」按鈕的結算頁會使用獨立的 `control.taps.result_confirm = [0.50, 0.93]`，直接點擊按鈕中央；沒有按鈕的結算頁才使用原圓形按鈕位置 `control.taps.result = [0.50, 0.80]`。

`output.save_debug_on_state_change` 與 `save_debug_on_action` 預設關閉。即使手動打開，只要偵測到魚已釣上並進入 `RESULT`，`cleanup_debug_each_round` 會立刻永久刪除 `debug` 內該輪所有 PNG/JPG，並禁止為結算狀態另存新圖；下一輪開始及程式停止時也會再次清理。UI 的「單次畫面辨識」固定覆寫 `debug/current.png`，且會在釣上魚時一併刪除。

調整後先重跑離線測試與單次診斷，不要直接開啟點擊。

## 測試

```powershell
python -m unittest discover -s tests -v
```

若需要重建模板：

```powershell
python tools\build_templates.py
```

## 限制

- 8 張參考圖不足以證明每輪完整轉場、每輪判定次數及結算關閉方式。
- H.264 串流使用 450 × 800 低延遲畫面，點擊座標則會另外讀取裝置實際解析度，避免縮放造成錯點。
- 自動化是否符合遊戲規範需由使用者自行確認。
