"""
core.py —— 心理状态分析核心逻辑

- 表情 → 心理状态映射
- 情绪累积追踪
- 会话状态管理
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import EMOTION_TO_PSYCH


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
#  会话状态
# ============================================================

@dataclass
class SessionState:
    history: list[dict[str, str]] = field(default_factory=list)
    emotion_log: list[str] = field(default_factory=list)       # 最近 30 次表情
    emotion_bucket: dict[str, float] = field(default_factory=dict)  # 各表情累计次数
    dominant_emotion: str = "中性"                               # 当前主导表情
    psych_label: str = "平静"                                    # 当前心理状态
    psych_strategy: str = "保持陪伴，不打扰"                      # 疏导策略
    session_start: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    # 疏导追踪
    last_counsel_time: float = 0.0                              # 上次疏导时间戳
    last_counseling: str = ""                                    # 上次疏导文案
    last_counseling_time_str: str = ""                           # 上次疏导时间（可读）
    counsel_count: int = 0                                       # 疏导次数
    # 最近帧信息
    last_emotion: str = "中性"                                    # 最近一帧的表情
    last_has_face: bool = False                                   # 最近一帧是否有人脸
    last_frame: object = None                                     # 最近一帧图像（供按钮使用）
    last_audio_path: str | None = None                            # 最近一次音频路径（避免刷新打断播放）
    last_audio_set_time: float = 0.0                               # 最近一次设置音频的时间戳（防重复触发）
    # 暂停 + 快照
    paused: bool = False                                            # 暂停守护
    last_snapshot_time: float = 0.0                                 # 上次拍照分析时间
    llm_thinking: bool = False                                      # LLM 正在思考中（过渡动画）
    _last_audio_sent: str = ""                                     # 上次发送到前端的音频路径（防重复播报）


# ============================================================
#  心理状态分类
# ============================================================

def classify_psychological_state(emotion: str) -> dict[str, str]:
    """
    根据表情判断心理状态。

    映射规则（可调整）：
        微笑 → 开心
        悲伤 → 情绪低落
        生气 → 焦虑
        害怕 → 焦虑
        惊讶 → 平静
        厌恶 → 情绪低落
        中性 → 平静
    """
    mapping = EMOTION_TO_PSYCH
    if emotion in mapping:
        return mapping[emotion]
    return {"label": "平静", "strategy": "保持陪伴，不打扰"}


# ============================================================
#  情绪累积与主导情绪
# ============================================================

def update_emotion_log(state: SessionState, emotion: str) -> SessionState:
    """更新情绪日志（滑动窗口 30 帧）。"""
    state.emotion_log.append(emotion)
    if len(state.emotion_log) > 30:
        state.emotion_log = state.emotion_log[-30:]

    # 统计各表情频率
    bucket: dict[str, float] = {}
    for e in state.emotion_log:
        bucket[e] = bucket.get(e, 0) + 1
    total = len(state.emotion_log)
    state.emotion_bucket = {k: v / total for k, v in bucket.items()}

    # 主导情绪
    if state.emotion_bucket:
        state.dominant_emotion = max(state.emotion_bucket, key=state.emotion_bucket.get)
    else:
        state.dominant_emotion = "中性"

    return state


def build_emotion_summary(state: SessionState) -> str:
    """构建近期的情绪摘要文本。"""
    if not state.emotion_bucket:
        return "暂无情绪数据。"

    items = sorted(state.emotion_bucket.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{emo}({pct:.0%})" for emo, pct in items[:3]]
    return "近期情绪分布: " + " / ".join(parts)


# ============================================================
#  对话记录
# ============================================================

def update_chat_history(
    state: SessionState,
    user_text: str,
    reply: str,
    emotion: str,
    psych_label: str,
) -> SessionState:
    """记录一次对话。"""
    state.history.append({
        "time": now_text(),
        "emotion": emotion,
        "psych": psych_label,
        "user": user_text.strip() or "（静默）",
        "reply": reply,
    })
    # 保留最近 20 条
    if len(state.history) > 20:
        state.history = state.history[-20:]
    return state


def get_chatbot_history(state: SessionState) -> list[dict[str, str]]:
    """转换为 Gradio Chatbot 格式。"""
    chat: list[dict[str, str]] = []
    for h in state.history[-10:]:
        chat.append({"role": "user", "content": h["user"]})
        chat.append({"role": "assistant", "content": h["reply"]})
    return chat
