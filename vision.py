"""
vision.py —— 使用 MediaPipe Face Landmarker 检测人脸 + 表情识别

- 加载 face_landmarker.task 模型
- 从 478 个面部关键点中提取几何特征
- 从 blendshapes（52个表情系数）辅助判断
- 输出 7 种表情：微笑、悲伤、生气、惊讶、害怕、厌恶、中性
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_tasks = None
    mp_vision = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


# ============================================================
#  MediaPipe 初始化
# ============================================================

@lru_cache(maxsize=1)
def _face_landmarker():
    """懒加载 + 缓存 FaceLandmarker 实例。"""
    if mp_vision is None or not MODEL_PATH.exists():
        return None
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=5,
        output_face_blendshapes=True,
    )
    try:
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        return exc


# ============================================================
#  几何特征计算
# ============================================================

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _detect_emotion_by_landmarks(
    landmarks: list, img_w: int, img_h: int
) -> tuple[str, dict[str, float]]:
    """基于 478 个面部关键点的几何关系判断表情。"""

    def p(i: int) -> tuple[float, float]:
        return (landmarks[i].x * img_w, landmarks[i].y * img_h)

    # ── 关键点索引 ──
    L_eye_outer, L_eye_inner = 33, 133
    L_eye_up,   L_eye_low   = 159, 145
    R_eye_outer, R_eye_inner = 362, 263
    R_eye_up,   R_eye_low   = 386, 374
    L_brow_inner, L_brow_outer = 46, 55
    R_brow_inner, R_brow_outer = 276, 285
    mouth_L, mouth_R = 61, 291
    mouth_up, mouth_low = 13, 14

    face_w = _dist(p(L_eye_outer), p(R_eye_outer))
    if face_w < 1:
        return "中性", {}

    # 嘴部特征
    mor = _dist(p(mouth_up), p(mouth_low)) / max(_dist(p(mouth_L), p(mouth_R)), 1)  # 张嘴程度
    mwr = _dist(p(mouth_L), p(mouth_R)) / face_w  # 嘴宽比
    mouth_cy = (p(mouth_up)[1] + p(mouth_low)[1]) / 2
    corner_avg_y = (p(mouth_L)[1] + p(mouth_R)[1]) / 2
    mcy = (corner_avg_y - mouth_cy) / face_w  # 嘴角上扬/下垂 (>0下垂 <0上扬)

    # 眼部特征
    L_ear = _dist(p(L_eye_up), p(L_eye_low)) / max(_dist(p(L_eye_outer), p(L_eye_inner)), 1)
    R_ear = _dist(p(R_eye_up), p(R_eye_low)) / max(_dist(p(R_eye_outer), p(R_eye_inner)), 1)

    # 眉毛特征
    L_brow_h = (p(L_brow_inner)[1] - p(L_eye_up)[1]) / face_w
    R_brow_h = (p(R_brow_inner)[1] - p(R_eye_up)[1]) / face_w
    L_brow_tilt = (p(L_brow_inner)[1] - p(L_brow_outer)[1]) / face_w
    R_brow_tilt = (p(R_brow_inner)[1] - p(R_brow_outer)[1]) / face_w

    features = {
        "mor": round(mor, 4), "mwr": round(mwr, 4), "mcy": round(mcy, 4),
        "L_ear": round(L_ear, 4), "R_ear": round(R_ear, 4),
        "L_brow_h": round(L_brow_h, 4), "R_brow_h": round(R_brow_h, 4),
        "L_brow_tilt": round(L_brow_tilt, 4), "R_brow_tilt": round(R_brow_tilt, 4),
    }

    # ── 表情判断 ──
    # 微笑：嘴角上扬 + 嘴巴变宽
    if mcy < -0.015 and mwr > 0.38:
        return "微笑", features
    if mcy < -0.02 and mwr > 0.35:
        return "微笑", features
    # 悲伤：嘴角下垂 或 眉毛上扬（内侧抬高=悲伤眉）
    if mcy > 0.02:
        return "悲伤", features
    if mcy > 0.01 and (L_brow_h < -0.015 or R_brow_h < -0.015):
        return "悲伤", features
    if (L_brow_h < -0.025 or R_brow_h < -0.025) and mwr < 0.35:
        return "悲伤", features
    # 惊讶：张嘴 + 抬眉
    if mor > 0.15 and (L_brow_h < -0.02 or R_brow_h < -0.02):
        return "惊讶", features
    if mor > 0.25:
        return "惊讶", features
    # 生气：眉压低 + 内侧下压
    if (L_brow_h > 0.01 or R_brow_h > 0.01) and (L_brow_tilt > 0.01 or R_brow_tilt > 0.01):
        return "生气", features
    # 害怕：抬眉 + 睁眼
    if (L_brow_h < -0.015 or R_brow_h < -0.015) and (L_ear > 0.28 or R_ear > 0.28):
        return "害怕", features
    # 厌恶：眉压低 + 嘴微张
    if (L_brow_h > 0.01 or R_brow_h > 0.01) and mor > 0.05:
        return "厌恶", features

    return "中性", features


# ============================================================
#  主检测函数
# ============================================================

def detect_frame_emotion(frame_rgb: np.ndarray | None) -> dict[str, Any]:
    """
    对一帧 RGB 图像进行人脸表情检测。

    返回:
        {
            "emotion": str,        # 微笑/悲伤/生气/惊讶/害怕/厌恶/中性
            "scores": dict,        # blendshape 分数 (52 个)
            "geom": dict,          # 几何特征
            "has_face": bool,      # 是否检测到人脸
            "mp_debug": str,       # 调试信息
        }
    """
    result: dict[str, Any] = {
        "emotion": "中性",
        "scores": {},
        "geom": {},
        "has_face": False,
        "mp_debug": "",
    }

    if frame_rgb is None:
        result["mp_debug"] = "frame is None"
        return result

    if mp is None or mp_vision is None:
        result["mp_debug"] = "mediapipe 未安装"
        return result

    if not MODEL_PATH.exists():
        result["mp_debug"] = f"模型文件缺失: {MODEL_PATH}"
        return result

    landmarker = _face_landmarker()
    if landmarker is None:
        result["mp_debug"] = "landmarker 初始化失败"
        return result
    if isinstance(landmarker, Exception):
        result["mp_debug"] = f"landmarker 错误: {landmarker}"
        return result

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection = landmarker.detect(mp_image)
        h, w = frame_rgb.shape[:2]

        if not detection.face_landmarks:
            result["mp_debug"] = "未检测到人脸"
            return result

        # 多人脸 → 社交场景
        face_count = len(detection.face_landmarks)
        if face_count > 1:
            result["has_face"] = True
            result["emotion"] = "多人"
            result["geom"] = {"face_count": face_count}
            result["mp_debug"] = f"MediaPipe: 多人({face_count}张脸)"
            return result

        result["has_face"] = True
        landmarks = detection.face_landmarks[0]

        # 几何特征判断
        emotion, geom = _detect_emotion_by_landmarks(landmarks, w, h)
        result["emotion"] = emotion
        result["geom"] = geom

        # Blendshape 分数（辅助参考）
        if detection.face_blendshapes:
            result["scores"] = {
                item.category_name: float(item.score)
                for item in detection.face_blendshapes[0]
            }

        result["mp_debug"] = f"MediaPipe: {emotion}"

    except Exception as exc:
        result["mp_debug"] = f"检测异常: {exc}"

    return result


# ============================================================
#  画面渲染
# ============================================================

def render_overlay(
    frame_rgb: np.ndarray | None,
    emotion: str,
    psych_label: str,
) -> np.ndarray:
    """在画面上叠加表情+心理状态信息。"""
    if frame_rgb is None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (15, 23, 42)
        return canvas

    if cv2 is None:
        return frame_rgb

    canvas = frame_rgb.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    h, w = canvas.shape[:2]

    # 顶部信息栏
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (10, 16, 24), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # 情绪颜色映射
    color_map = {
        "微笑": (80, 230, 100),
        "悲伤": (100, 150, 255),
        "生气": (100, 100, 240),
        "惊讶": (200, 180, 100),
        "害怕": (180, 120, 220),
        "厌恶": (120, 200, 100),
        "中性": (200, 200, 200),
    }
    color = color_map.get(emotion, (200, 200, 200))

    cv2.putText(canvas, f"表情: {emotion}", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"心理: {psych_label}", (24, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 244, 214), 2, cv2.LINE_AA)

    return canvas


# ============================================================
#  独立运行：摄像头实时表情识别演示
# ============================================================

def main():
    """独立运行 vision.py：打开摄像头，实时显示表情识别结果。"""
    print("=" * 56)
    print("  🎭 人脸表情识别 · 独立演示")
    print("  使用 MediaPipe Face Landmarker（478 点 + 52 blendshapes）")
    print("  按 Q 或 ESC 退出，按 C 切换摄像头")
    print("=" * 56)

    if cv2 is None:
        print("❌ 缺少 opencv-python，无法显示画面。")
        sys.exit(1)

    cam_index = 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {cam_index}")
        sys.exit(1)

    print(f"📷 当前摄像头: {cam_index}")

    while True:
        success, frame_bgr = cap.read()
        if not success:
            print("读取摄像头帧失败。")
            break

        # 镜像翻转，更符合直觉
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 调用与主项目完全相同的检测函数
        result = detect_frame_emotion(frame_rgb)
        emotion = result.get("emotion", "?")
        scores = result.get("scores", {})
        mp_debug = result.get("mp_debug", "")
        geom = result.get("geom", {})

        # 画面叠加信息面板
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (10, 10), (440, 200), (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

        # 表情大字
        color_map = {
            "微笑": (80, 230, 100),
            "悲伤": (100, 150, 255),
            "生气": (100, 100, 240),
            "惊讶": (200, 180, 100),
            "害怕": (180, 120, 220),
            "厌恶": (120, 200, 100),
            "中性": (200, 200, 200),
            "多人": (230, 180, 60),
        }
        color = color_map.get(emotion, (200, 200, 200))
        cv2.putText(frame_bgr, f"Emotion: {emotion}  [Cam {cam_index}]",
                    (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # 关键表情系数条形图
        y0, dy = 82, 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        row = 0
        for name in ["mouthSmileLeft", "browDownLeft", "jawOpen", "eyeBlinkLeft", "eyeSquintLeft"]:
            s = scores.get(name, 0)
            bar = "#" * int(s * 20)
            cv2.putText(frame_bgr, f"{name}: {bar} ({s:.2f})",
                        (28, y0 + dy * row), font, 0.42, (220, 230, 240), 1, cv2.LINE_AA)
            row += 1

        # 几何特征摘要
        if geom:
            geom_str = " ".join(f"{k}={v:.3f}" for k, v in list(geom.items())[:4])
            cv2.putText(frame_bgr, f"geom: {geom_str}", (28, y0 + dy * 6),
                        font, 0.4, (150, 200, 255), 1, cv2.LINE_AA)

        cv2.putText(frame_bgr, mp_debug, (28, y0 + dy * 7),
                    font, 0.4, (150, 200, 255), 1, cv2.LINE_AA)

        # 底部提示
        cv2.putText(frame_bgr, "C 切换摄像头  |  Q/ESC 退出  |  vision.py 独立运行",
                    (28, frame_bgr.shape[0] - 20), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Emotion Recognition (vision.py)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            cap.release()
            cam_index = (cam_index + 1) % 10
            cap = cv2.VideoCapture(cam_index)
            if cap.isOpened():
                print(f"📷 切换到摄像头 {cam_index}")
            else:
                print("⚠️  摄像头不可用，切回 0")
                cap.release()
                cam_index = 0
                cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(100)
    cv2.waitKey(1)
    print("已退出")


if __name__ == "__main__":
    main()
