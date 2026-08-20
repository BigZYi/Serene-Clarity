"""
llm.py —— LLM 心理疏导文案生成

- 支持 DeepSeek API 远程调用
- 支持本地规则降级（DEBUG_MODE）
- 时间感知（深夜/清晨/下午不同回应风格）
- 结合表情、心理状态、天气信息生成疏导文案
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from urllib import request

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEBUG_MODE,
    WEATHER_CITY,
)
from weather_service import get_weather_context


# ============================================================
#  本地降级文案（不调 API 时使用）
# ============================================================

LOCAL_COUNSELING: dict[str, str] = {
    "开心": "看到你笑了，真好！这种时刻值得被记住——可以写一句话记下来，以后回头看会觉得温暖。",
    "情绪低落": "总有这样的时刻。不必急着走出来，可以先让它在。有时候天气不好，心情也跟着低落，这很正常。",
    "焦虑": "先停一下。3 秒吸气、4 秒吐气。作业、考试、未来……事情可以一件一件来，你不用同时处理所有。",
    "平静": "平静本身就是一种力量。保持住，今天你做得很好。",
    "陪伴": "和朋友在一起呢，真好。享受这段时光吧。",
}


# ============================================================
#  时间感知
# ============================================================

def _time_context() -> str:
    """根据当前时间返回时段描述 + 回应风格提示。"""
    hour = datetime.now().hour
    if 0 <= hour < 6:
        return "现在是深夜。你的语气要特别温柔，提醒早点休息、别熬夜。"
    elif 6 <= hour < 9:
        return "现在是清晨。你的语气要精神一点，鼓励迎接新的一天。"
    elif 9 <= hour < 12:
        return "现在是上午。可以关心在忙什么，提醒别忘了吃早饭。"
    elif 12 <= hour < 14:
        return "现在是中午。提醒按时吃午饭，别总对付。"
    elif 14 <= hour < 18:
        return "现在是下午。语气轻松一些，可以聊聊今天过得怎么样。"
    elif 18 <= hour < 22:
        return "现在是晚上。语气温暖，关心晚上吃好了没，别太累。"
    else:
        return "现在是深夜。语气轻声细语，像睡前最后的叮咛，催早点睡。"


# ============================================================
#  LLM 提示词构建
# ============================================================

def build_system_prompt(
    psych_label: str,
    psych_strategy: str,
    emotion: str,
    emotion_summary: str,
    user_text: str = "",
) -> str:
    """构建发送给 LLM 的系统提示词。"""
    weather_ctx = (
        get_weather_context(WEATHER_CITY) if WEATHER_CITY else get_weather_context()
    )
    time_ctx = _time_context()

    # 表情 → 情绪线索（用于引导语气）
    emotion_hint = {
        "悲伤": "对方情绪有些低落，先共情、慢慢陪，别急着讲道理。",
        "微笑": "对方心情不错，可以温和地回应这份好心情。",
        "生气": "对方可能有烦心事，语气要平缓，先让他把情绪稳住。",
        "害怕": "对方有些不安，给予安全感、肯定他在。",
        "厌恶": "对方有些抵触或疲惫，语气放轻，不追问。",
        "惊讶": "对方似乎遇到新鲜事，保持平静自然的陪伴。",
        "多人": "对方正和朋友在一起，简单轻松地打个招呼就好。",
        "中性": "对方看起来很平静，用日常轻松的口吻说两句即可。",
    }.get(emotion, "像平时一样温和自然地说话。")

    return (
        "你是一位桌面 AI 情绪管理助手，负责根据对方的面部表情、当前心理状态和实时天气，"
        "输出简短、温暖、有分寸的中文心理疏导。\n"
        "你的定位：不是长辈、不是医生、不是心理咨询师，而是一个懂分寸、会倾听的 AI 同伴。\n"
        "对方是在校大学生，可能面临学业压力、人际关系、想家、迷茫等常见情绪。\n"
        "\n【原则】\n"
        "- 不说教，不假装共情，不用「你应该」，不居高临下。\n"
        "- 用简洁自然的中文，1~2 句话，像朋友聊天一样。\n"
        "- 内容要落地：可以结合天气给出具体的小建议（降温提醒穿衣、下雨带伞、好天气出去走走）。\n"
        "- 绝不提到「摄像头」「监控」「检测」「识别」「AI」「模型」等词，像天然的人在说话。\n"
        f"\n── 当前状态 ──\n"
        f"表情：{emotion}（{emotion_hint}）\n"
        f"心理状态：{psych_label}\n"
        f"疏导策略：{psych_strategy}\n"
        f"{time_ctx}\n"
        f"\n── 近期情绪 ──\n"
        f"{emotion_summary}\n"
        f"\n── 当前环境 ──\n"
        f"{weather_ctx or '天气数据暂不可用。'}\n"
        f"\n回复要求：1~2 句，自然平和，不超过 120 字。"
        f"根据时间调整语气（深夜温柔、白天精神），结合天气给出具体提醒。"
    )


# ============================================================
#  主函数
# ============================================================

def generate_counseling(
    psych_label: str,
    psych_strategy: str,
    emotion: str,
    emotion_summary: str,
    user_text: str = "",
) -> str:
    """
    生成心理疏导文案。

    本地模式 (DEBUG_MODE=True)：使用预置文案。
    远程模式：调用 DeepSeek API。
    """
    # ── 本地降级 ──
    if DEBUG_MODE:
        return LOCAL_COUNSELING.get(psych_label, "我在听。你慢慢说。")

    # ── 远程调用 ──
    api_key = (os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY or "").strip()
    base_url = (os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL or "https://api.deepseek.com").strip()
    model = (os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_MODEL or "deepseek-v4-pro").strip()

    if not api_key:
        return LOCAL_COUNSELING.get(psych_label, "我在听。你慢慢说。")

    system_prompt = build_system_prompt(psych_label, psych_strategy, emotion, emotion_summary, user_text)
    user_content = f"用户状态：{user_text.strip() or '（用户未说话）'}\n请给出简短的心理疏导。"

    try:
        return _call_deepseek(api_key, model, system_prompt, user_content, base_url)
    except Exception as exc:
        fallback = LOCAL_COUNSELING.get(psych_label, "我在听。你慢慢说。")
        print(f"[LLM] API 调用失败，使用本地降级: {exc}")
        return fallback


def _call_deepseek(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    base_url: str = "https://api.deepseek.com",
    timeout: int = 20,
) -> str:
    """底层 DeepSeek Chat API 调用。"""
    base = base_url.strip().rstrip("/")
    url = f"{base}/chat/completions"

    payload = {
        "model": model or "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 220,
        "stream": False,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    body = json.loads(raw)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek 返回空响应")

    content = (choices[0].get("message") or {}).get("content", "").strip()
    if not content:
        raise RuntimeError("DeepSeek 返回空内容")

    return content
