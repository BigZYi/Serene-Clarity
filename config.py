"""
配置中心 —— 人脸表情 → 心理状态 → 天气 → LLM 心理疏导
=====================================================
所有可配置项集中于此。修改后重启生效。
敏感信息也可用同名环境变量覆盖。
"""

from __future__ import annotations

# ─── DeepSeek LLM ──────────────────────────────────────
# 也可用环境变量 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL 覆盖
DEEPSEEK_API_KEY = "#"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ─── 天气 ─────────────────────────────────────────────
WEATHER_CITY = ""   # 留空 = IP 自动定位；填城市名 = 固定城市

# ─── 应用设置 ─────────────────────────────────────────
APP_TITLE = "心晴 · AI 心理疏导"
APP_PORT = 7860
DEBUG_MODE = False  # True: 本地规则回复（不调 API）；False: 调用 DeepSeek
LLM_INTERVAL_SECONDS = 30  # 每隔多少秒进行一次情绪汇总并调用 LLM
SNAPSHOT_INTERVAL = 5        # 拍立得模式：每隔多少秒拍一张照片分析（减少流式压迫感）

# ─── 语音播报（TTS）───────────────────────────────────
TTS_ENABLED = True            # 是否开启心理疏导语音播报
VOICE_NAME = "my_voice_1785394957"  # 已注册的音色

# ─── 华为云 VCS（声音合成）──────────────────────────
HUAWEI_AK = "HPUAECYCEZPEWS2YIMZ1"
HUAWEI_SK = "LhWVRfAAZLSdCdEm6A7fiuwUg4X77To418OWsD9l"
HUAWEI_REGION = "cn-east-3"
HUAWEI_PROJECT_ID = "5860acea06914e2ab9351d92286c4677"

# ─── 表情 → 心理状态映射 ─────────────────────────────
EMOTION_TO_PSYCH = {
    "微笑":  {"label": "开心",     "strategy": "强化积极体验，建议记录下来"},
    "悲伤":  {"label": "情绪低落", "strategy": "先共情，不急于解决"},
    "生气":  {"label": "焦虑",     "strategy": "引导深呼吸 + 拆分问题"},
    "惊讶":  {"label": "平静",     "strategy": "保持陪伴，不打扰"},
    "害怕":  {"label": "焦虑",     "strategy": "确认安全感 + 引导放松"},
    "厌恶":  {"label": "情绪低落", "strategy": "接纳感受，温和引导"},
    "中性":  {"label": "平静",     "strategy": "保持陪伴，不打扰"},
    "多人":  {"label": "陪伴",     "strategy": "你在和朋友一起，真好"},
}

# ─── 心理状态 → 风格化色调 ────────────────────────────
PSYCH_COLORS = {
    "开心":     "#10b981",  # 绿
    "情绪低落": "#6366f1",  # 靛
    "焦虑":     "#f59e0b",  # 琥珀
    "平静":     "#3b82f6",  # 蓝
    "陪伴":     "#ec4899",  # 粉
}
