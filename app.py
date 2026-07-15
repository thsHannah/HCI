# -*- coding: utf-8 -*-
"""
FLOWDRAW - Python 版本 (Flask + OpenCV + MediaPipe)

架構說明：
  - 攝影機擷取與手部追蹤、手勢判斷、筆劃繪製全部在「伺服器端」用 Python 完成
    （OpenCV 抓攝影機畫面 + MediaPipe **Tasks API** 的 HandLandmarker 做手部關鍵點偵測）
  - 處理完的畫面用 MJPEG (multipart/x-mixed-replace) 的方式串流給瀏覽器顯示
  - 瀏覽器端的按鈕（清除、筆刷大小、顏色、下載）改成呼叫後端的 REST API

⚠️ 關於 mediapipe 版本：
  第一次執行時程式會自動從 Google 官方下載一個約 8MB 的模型檔到
  `models/hand_landmarker.task`（需要網路），之後就會直接用快取檔，
  不會重複下載。如果上課現場沒有網路，記得**提前在有網路的地方跑過一次**，
  讓每台學生電腦都先把模型檔快取起來。

⚠️ 重要限制：
  cv2.VideoCapture(0) 抓的是「執行這支 Python 程式的那台機器」的攝影機，
  不是瀏覽器使用者的攝影機。所以這個架構只適合：
    1. 在自己電腦上跑 `python app.py`，然後用瀏覽器打開 http://127.0.0.1:5000
    2. 或是部署在一台有攝影機、且你想讓所有連進來的人共用同一支攝影機畫面的機器上
  如果你要「每個瀏覽器使用者用自己的攝影機」，那必須改用 WebRTC 把畫面從
  瀏覽器傳到後端（複雜很多），這份程式碼沒有做這件事。
"""

import io
import os
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from flask import Flask, Response, request, jsonify, render_template, send_file

# ----------------------------------------------------------------------------
# `fingers` 常數表
# ----------------------------------------------------------------------------
FINGERS = {
    "index1": 8, "index2": 7, "index3": 6, "index4": 5,
    "middle1": 12, "middle2": 11, "middle3": 10, "middle4": 9,
    "ring1": 16, "ring2": 15, "ring3": 14, "ring4": 13,
    "little1": 20, "little2": 19, "little3": 18, "little4": 17,
    "thumb1": 4, "thumb2": 3, "thumb3": 2, "thumb4": 1, "thumb5": 0,
}

ERASE_RADIUS = 40

# ----------------------------------------------------------------------------
# mediapipe Tasks API 用的 21 個手部關鍵點連線（畫骨架用），
# 對應舊版 mp.solutions.hands.HAND_CONNECTIONS 的內容（新版套件已不再提供這個常數）
# ----------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 大拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (0, 17),                                  # 手掌
]

# Google 官方提供的 HandLandmarker 模型檔下載位置
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)


def ensure_hand_landmarker_model(path=HAND_LANDMARKER_MODEL_PATH, url=HAND_LANDMARKER_MODEL_URL):
    """
    第一次執行時自動下載 HandLandmarker 模型檔（約 8MB），之後會快取在 models/ 資料夾，
    不會每次都重新下載。如果這台機器連不到網路，請在有網路的機器上先下載這個檔案，
    再手動複製到 models/hand_landmarker.task。

    回傳的是模型檔的「位元組內容 (bytes)」而不是路徑字串——mediapipe 在 Windows 上
    對含有非英文字元（例如中文資料夾名稱）的路徑開檔時，底層 C++ 有時會失敗並丟出
    RuntimeError: Unable to open file ... errno=-1/22。直接把內容讀成 bytes 餵給
    mediapipe，可以完全避開這個路徑相關的問題。
    """
    need_download = True
    if os.path.exists(path):
        # 檔案存在，但也要檢查大小，避免之前下載到一半失敗、留下一個壞掉的殘檔
        if os.path.getsize(path) > 1_000_000:  # 正常模型檔約 7~9MB，小於 1MB 視為壞檔
            need_download = False
        else:
            print(f"[setup] 偵測到 {path} 檔案大小異常（可能是上次下載失敗的殘檔），重新下載", flush=True)
            os.remove(path)

    if need_download:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[setup] 第一次執行，正在下載手部偵測模型到 {path} ...", flush=True)
        urllib.request.urlretrieve(url, path)
        print("[setup] 模型下載完成", flush=True)

    with open(path, "rb") as f:
        return f.read()


# ----------------------------------------------------------------------------
# 幾何工具函式，對應 vector2DAngle() / calculateFingerAngles() / isFist()
# ----------------------------------------------------------------------------
def vector2d_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_finger_angles(landmarks):
    """landmarks 是 mediapipe 回傳的 21 個關鍵點 (有 .x .y)"""
    triples = [
        ("thumb1", "thumb2", "thumb3"),
        ("index1", "index2", "index3"),
        ("middle1", "middle2", "middle3"),
        ("ring1", "ring2", "ring3"),
        ("little1", "little2", "little3"),
    ]
    angles = []
    for a, b, c in triples:
        la, lb, lc = landmarks[FINGERS[a]], landmarks[FINGERS[b]], landmarks[FINGERS[c]]
        v1 = [lb.x - la.x, lb.y - la.y]
        v2 = [lc.x - lb.x, lc.y - lb.y]
        angles.append(vector2d_angle(v1, v2))
    return angles


def is_fist(angles, threshold=50):
    return all(a >= threshold for a in angles)


# ----------------------------------------------------------------------------
# Point / StrokeList，對應 strokes.js 的 class Point / class StrokeList
# ----------------------------------------------------------------------------
class Point:
    __slots__ = ("x", "y", "size", "color")

    def __init__(self, x, y, size=5, color=(0, 0, 255)):
        self.x = x
        self.y = y
        self.size = size
        self.color = color  # BGR tuple，OpenCV 用 BGR

    @staticmethod
    def distance(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


class StrokeList:
    def __init__(self):
        self.stroke_list = [[]]
        self.brush_size = 5
        self.color = (0, 0, 255)  # 預設紅色 (BGR)
        self._lock = threading.Lock()

    def increase_brush_size(self):
        with self._lock:
            self.brush_size += 5

    def decrease_brush_size(self):
        with self._lock:
            if self.brush_size > 5:
                self.brush_size -= 5

    def change_brush_color(self, bgr_color):
        with self._lock:
            self.color = bgr_color

    def add_pt(self, x, y):
        with self._lock:
            # ✨ 安全檢查：如果 stroke_list 完全是空的，先補一個空筆劃給它
            if not self.stroke_list:
                self.stroke_list.append([])

            pt = Point(x, y, self.brush_size, self.color)
            self.stroke_list[-1].append(pt)

    def clear(self):
        with self._lock:
            self.stroke_list = [[]]  # 確保清除後裡面有一個空筆劃，而不是完全空的 []

    def erase(self, erase_pos, radius):
        with self._lock:
            new_strokes = []
            for stroke in self.stroke_list:
                current = []
                for pt in stroke:
                    if Point.distance(erase_pos, pt) > radius + pt.size / 2:
                        current.append(pt)
                    elif current:
                        new_strokes.append(current)
                        current = []
                if current:
                    new_strokes.append(current)
            self.stroke_list = [s for s in new_strokes if s]

    def new_stroke(self):
        with self._lock:
            if not self.stroke_list or self.stroke_list[-1]:
                self.stroke_list.append([])

    def draw(self, frame):
        with self._lock:
            strokes_snapshot = [list(s) for s in self.stroke_list]
        for stroke in strokes_snapshot:
            for i in range(1, len(stroke)):
                p0, p1 = stroke[i - 1], stroke[i]
                cv2.line(
                    frame,
                    (int(p0.x), int(p0.y)),
                    (int(p1.x), int(p1.y)),
                    p1.color,
                    int(p1.size),
                    lineType=cv2.LINE_AA,
                )


# ----------------------------------------------------------------------------
# gesture()：對應 scriptent.js 的 gesture()
#   0: 沒事 / 1: 食指(畫圖) / 2: 食指+中指(擦除) 
# ----------------------------------------------------------------------------
def gesture(finger_state):
    if finger_state["index"] and not finger_state["middle"] and not finger_state["ring"] and not finger_state["little"]:
        return 1
    if finger_state["index"] and finger_state["middle"] and not finger_state["ring"] and not finger_state["little"]:
        return 2
    # if finger_state["isFist"]:
    #     return 3
    # if finger_state["index"] and finger_state["middle"] and finger_state["ring"] and not finger_state["little"]:
    #     return 4
    return 0


# ----------------------------------------------------------------------------
# VideoProcessor：把 init()/process()/processHands() 的邏輯整合在這個類別裡，
# 用一個背景執行緒持續讀取攝影機畫面、跑手勢辨識、更新 stroke_list
# ----------------------------------------------------------------------------
class VideoProcessor:
    def __init__(self, camera_index=0, width=1280, height=720):
        self.width = width
        self.height = height

        print(f"[VideoProcessor] 開啟攝影機 index={camera_index} ...", flush=True)
        self.cap = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
        print(f"[VideoProcessor] 攝影機開啟完成，isOpened()={self.cap.isOpened()}", flush=True)
        if not self.cap.isOpened():
            print("[VideoProcessor] ⚠️ 攝影機沒有成功開啟！"
                  "請確認這台機器有攝影機、沒有被其他程式占用、"
                  "且已允許 Python/終端機存取攝影機權限。"
                  "可以先跑 `python camera_test.py` 單獨排查。", flush=True)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("[VideoProcessor] 載入 MediaPipe HandLandmarker 模型 ...", flush=True)
        model_bytes = ensure_hand_landmarker_model()
        base_options = mp_tasks_python.BaseOptions(model_asset_buffer=model_bytes)
        options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)
        print("[VideoProcessor] MediaPipe HandLandmarker 模型載入完成", flush=True)

        self.stroke_list = StrokeList()

        self.finger_state = {
            "landmarks": None,
            "index": False,
            "middle": False,
            "ring": False,
            "little": False,
            "isFist": False,
        }

        self.previous_pt_active = False
        self.save_cooldown = False
        self.background_cooldown = False
        self.bg_index = 0

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.last_gesture = 0
        self.last_saved_png = None  # 最近一次「握拳存檔」產生的 PNG bytes

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[VideoProcessor] 背景擷取執行緒已啟動", flush=True)

    # -- 對應 processHands()：改為新版 Tasks API 的 HandLandmarkerResult --
    def _process_hands(self, result, frame):
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # 只取第一隻手（num_hands=1）
            h, w = frame.shape[:2]

            # 手動畫骨架（新版套件已不再附贈 drawing_utils）
            for a, b in HAND_CONNECTIONS:
                pa, pb = lm[a], lm[b]
                cv2.line(frame, (int(pa.x * w), int(pa.y * h)),
                          (int(pb.x * w), int(pb.y * h)), (0, 255, 0), 2)
            for pt in lm:
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 0, 255), -1)

            self.finger_state["landmarks"] = lm
            self.finger_state["index"] = lm[FINGERS["index1"]].y < lm[FINGERS["index3"]].y
            self.finger_state["middle"] = lm[FINGERS["middle1"]].y < lm[FINGERS["middle3"]].y
            self.finger_state["ring"] = lm[FINGERS["ring1"]].y < lm[FINGERS["ring3"]].y
            self.finger_state["little"] = lm[FINGERS["little1"]].y < lm[FINGERS["little3"]].y
            angles = calculate_finger_angles(lm)
            self.finger_state["isFist"] = is_fist(angles)
        else:
            self.finger_state["landmarks"] = None
            self.finger_state["index"] = False
            self.finger_state["middle"] = False
            self.finger_state["ring"] = False
            self.finger_state["little"] = False
            self.finger_state["isFist"] = False

    # -- 畫面上顯示目前狀態，取代原本的 draw_icon/erase_icon/save_icon 疊圖 --
    def _draw_state_overlay(self, frame, gest):
        h, w = frame.shape[:2]
        label_map = {
            1: ("DRAWING", (0, 255, 0)),
            2: ("ERASING", (0, 165, 255)),
            3: ("SAVED!", (255, 0, 255)),
            4: ("SWITCH BG", (255, 255, 0)),
        }
        if gest in label_map:
            text, color = label_map[gest]
            cv2.rectangle(frame, (w - 220, h - 60), (w - 10, h - 15), (30, 30, 30), -1)
            cv2.putText(frame, text, (w - 205, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
        # 目前筆刷顏色/大小小色塊（左上角）
        cv2.circle(frame, (30, 30), self.stroke_list.brush_size + 5,
                   self.stroke_list.color, -1)

    # -- 對應 process() 主迴圈 --------------------------------------------
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)  # 鏡像，符合直覺（原本 JS selfieMode:false，這裡依需求可拿掉 flip）

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            self._process_hands(result, frame)
            gest = gesture(self.finger_state)
            self.last_gesture = gest

            # --- 背景切換 (手勢 4)：這裡先只保留冷卻邏輯，實際背景圖片可自行擴充 ---
            # if gest == 4 and not self.background_cooldown:
            #     self.bg_index += 1
            #     self.background_cooldown = True
            #     threading.Timer(3.0, self._reset_bg_cooldown).start()

            # --- 畫圖 (手勢 1) ---
            # 注意：開新筆劃的時機是「重新開始畫」的那一瞬間，而不是「停止畫」的那一瞬間。
            # 如果在「停止畫」時就預先塞一個空筆劃當佔位，中間只要做過擦除手勢，
            # erase() 會把所有空筆劃過濾掉，導致佔位筆劃被清掉；等下次真正開始畫圖時，
            # add_pt() 就會接到「上一次畫圖結束時的舊筆劃」尾端，造成新的一筆線從上次
            # 結束的位置直接連過去（看起來像是起始位置停留在上次結束的地方）。
            if gest == 1 and self.finger_state["landmarks"] is not None:
                idx = self.finger_state["landmarks"][FINGERS["index1"]]
                if not self.previous_pt_active:
                    self.stroke_list.new_stroke()
                self.stroke_list.add_pt(idx.x * self.width, idx.y * self.height)
                self.previous_pt_active = True
            else:
                self.previous_pt_active = False

            # --- 擦除 (手勢 2) ---
            if gest == 2 and self.finger_state["landmarks"] is not None:
                idx = self.finger_state["landmarks"][FINGERS["index1"]]
                mdl = self.finger_state["landmarks"][FINGERS["middle1"]]
                erase_pos = Point(
                    self.width * (idx.x + mdl.x) / 2.0,
                    self.height * (idx.y + mdl.y) / 2.0,
                )
                self.stroke_list.erase(erase_pos, ERASE_RADIUS)
                cv2.circle(frame, (int(erase_pos.x), int(erase_pos.y)),
                           ERASE_RADIUS, (114, 128, 250), 3)

            # --- 握拳存檔 (手勢 3) ---
            # 這裡只是把當下畫面(含筆劃)先編碼成 PNG、暫存在伺服器記憶體裡。
            # 要讓瀏覽器真正下載到本機，還是要按下頁面上的「Download Image」按鈕，
            # 因為 MJPEG 串流架構下，伺服器無法主動觸發瀏覽器下載檔案。
            # 1. 先將所有筆劃與疊圖畫在 frame 上
            #self.stroke_list.draw(frame)
            #self._draw_state_overlay(frame, gest)

            # 2. 畫完之後，立刻更新最新畫面 (最新畫面現在有筆劃了！)
            #with self.frame_lock:
            #    self.latest_frame = frame

            # 3. 此時如果偵測到握拳 (手勢 3)，拿這個「已經畫好筆劃」的 frame 去存檔！
            # if gest == 3 and not self.save_cooldown:
            #     self.save_cooldown = True
            #     self._save_snapshot(frame)  # 這裡傳入的 frame 已經包含完整的筆痕了！
            #     threading.Timer(2.0, self._reset_save_cooldown).start()
            time.sleep(1 / 30)

    def _reset_bg_cooldown(self):
        self.background_cooldown = False

    def _reset_save_cooldown(self):
        self.save_cooldown = False

    # -- 對應 saveCanvasAsImage()：把目前畫面(含筆劃)編碼成 PNG bytes 存起來 --
    def _save_snapshot(self, frame):
        ok, buf = cv2.imencode(".png", frame)
        if ok:
            self.last_saved_png = buf.tobytes()

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self.latest_frame)
            if not ok:
                return None
            return buf.tobytes()

    def stop(self):
        self.running = False
        self.cap.release()
        self.landmarker.close()


# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__)
processor = VideoProcessor()


def mjpeg_generator():
    while True:
        frame_bytes = processor.get_jpeg_frame()
        if frame_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(1 / 30)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify({
        "brush_size": processor.stroke_list.brush_size,
        "color_bgr": processor.stroke_list.color,
        "gesture": processor.last_gesture,
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    processor.stroke_list.clear()
    return jsonify({"ok": True})


@app.route("/api/brush/increase", methods=["POST"])
def api_brush_increase():
    processor.stroke_list.increase_brush_size()
    return jsonify({"ok": True, "brush_size": processor.stroke_list.brush_size})


@app.route("/api/brush/decrease", methods=["POST"])
def api_brush_decrease():
    processor.stroke_list.decrease_brush_size()
    return jsonify({"ok": True, "brush_size": processor.stroke_list.brush_size})


@app.route("/api/color", methods=["POST"])
def api_color():
    """
    前端傳入 JSON: { "r": 255, "g": 0, "b": 0 }
    OpenCV 用 BGR，所以這裡轉換一下順序
    """
    data = request.get_json(force=True)
    r, g, b = int(data.get("r", 255)), int(data.get("g", 0)), int(data.get("b", 0))
    processor.stroke_list.change_brush_color((b, g, r))
    return jsonify({"ok": True})


@app.route("/api/download_image")
def api_download_image():
    """下載最近一次握拳觸發、或即時當前畫面存成的 PNG"""
    if processor.last_saved_png is None:
        # 沒有握拳存過檔，就直接抓目前畫面即時編碼一張
        with processor.frame_lock:
            frame = processor.latest_frame
        if frame is None:
            return jsonify({"ok": False, "error": "no frame yet"}), 404
        ok, buf = cv2.imencode(".png", frame)
        png_bytes = buf.tobytes() if ok else None
    else:
        png_bytes = processor.last_saved_png

    if png_bytes is None:
        return jsonify({"ok": False}), 500

    filename = f"drawing_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
    return send_file(io.BytesIO(png_bytes), mimetype="image/png",
                      as_attachment=True, download_name=filename)


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
    finally:
        processor.stop()