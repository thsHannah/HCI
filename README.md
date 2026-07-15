# FLOWDRAW - Python 後端版

原本的 `scriptent.js` + `strokes.js` 已經改寫成 Python，放在 `app.py` 裡，
用 **Flask + OpenCV + MediaPipe** 在伺服器端做攝影機擷取、手部追蹤、手勢判斷與畫圖，
再把處理好的畫面以 MJPEG 串流方式送到瀏覽器的 `<img>` 標籤顯示。

##  關於 Python 版本與 mediapipe API（給老師/助教看）

**第一次執行需要網路**：程式會自動從 Google 官方下載一個約 8MB 的手部偵測模型檔到
`models/hand_landmarker.task`，下載一次後就會快取，之後離線也能跑。
如果上課現場網路不穩，**建議提前在每台學生電腦上先跑過一次 `python app.py`**，
確認 `models/hand_landmarker.task` 已經下載成功，再帶去上課。
也可以在一台電腦上下載好後，把整個 `models/` 資料夾複製給其他台電腦，不用每台都重新下載。

## 安裝

```bash
python -m venv venv
source venv/bin/activate      # Windows 用 venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

```bash
python app.py
```

然後用瀏覽器打開 **http://127.0.0.1:5000**

## 手勢對照

| 手勢 | 動作 |
|---|---|
| 只有食指伸直 | 畫圖 |
| 食指 + 中指伸直 | 擦除 |

## 檔案對照表

| 原本 JS | Python |
|---|---|
| `fingers` 常數 | `FINGERS` dict |
| `vector2DAngle()` | `vector2d_angle()` |
| `calculateFingerAngles()` | `calculate_finger_angles()` |
| `gesture()` | `gesture()` |
| `class Point` | `class Point` |
| `class StrokeList` | `class StrokeList` |
| `saveCanvasAsImage()` | `VideoProcessor._save_snapshot()` + `/api/download_image` |
| `download_points()` | `/api/download_points` |
| `init()` / `process()` / `processHands()` | `VideoProcessor` 類別 + 背景執行緒 `_loop()`（手部偵測用新版 Tasks API `HandLandmarker`，非舊版 `mp.solutions.hands`）|

## 重要限制

1. **攝影機來源不同**：原本 JS 版用瀏覽器的 `getUserMedia()`，抓的是「使用者自己」的攝影機，
   畫面完全不會離開使用者電腦。這個 Python 版改成 `cv2.VideoCapture(0)`，抓的是
   **執行 `app.py` 這台機器**的攝影機。也就是說：
   - 如果你在自己筆電上跑，等於還是你自己的攝影機，沒問題。
   - 如果你把這支程式部署到遠端伺服器，那伺服器需要接攝影機，而且畫面只有一份，
     所有連進來的瀏覽器看到的是同一支攝影機、同一份畫布狀態（沒有做多使用者隔離）。
   - 如果你要「每個瀏覽器使用者用自己的攝影機」，需要改用 WebRTC 把瀏覽器端影像串流
     送進後端處理（架構複雜很多，這份程式碼沒有實作）。

2. **圖示資源沒有搬過來**：原本的 `assets/draw.png`、`erase.png`、`fist.png`、
   `background1.jpg` 這幾個檔案沒有附在對話裡，所以 Python 版改用畫面右下角的文字
   徽章（DRAWING / ERASING / SAVED! / SWITCH BG）取代圖示。如果你有這些圖片，
   可以把它們放進 `static/assets/`，然後在 `VideoProcessor._draw_state_overlay()`
   裡用 `cv2.imread` 讀進來、疊到 frame 上（有留註解位置）。

3. **效能**：MediaPipe Python 版在 CPU 上跑，速度會比瀏覽器裡跑 TF.js/WASM 版本慢一些，
   如果延遲明顯，可以調低 `VideoProcessor(width=..., height=...)`的解析度，
   或把 `model_complexity` 從 1 改成 0。