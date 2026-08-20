"""
broadcast.py —— 语音播报：将心理疏导文案合成为语音

使用华为云 VCS（声音克隆/语音合成）将 LLM 生成的疏导文案播报出来，
返回音频文件路径供 Gradio 播放。

前置条件：
    1. config.py 中 VOICE_NAME 已填写已注册的音色名
    2. 已配置 HUAWEI_AK / HUAWEI_SK / HUAWEI_REGION / HUAWEI_PROJECT_ID
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from config import (
    HUAWEI_AK,
    HUAWEI_REGION,
    HUAWEI_PROJECT_ID,
    HUAWEI_SK,
    TTS_ENABLED,
    VOICE_NAME,
)

# 华为云 SDK 路径（在 speech_llms 子目录下）
_HUAWEI_SDK = Path(__file__).resolve().parent.parent / "Day2" / \
    "2.LLM认知与调用、Gradio简易交互设计" / "Gradio-LLM_stu" / "speech_llms"
if str(_HUAWEI_SDK) not in sys.path:
    sys.path.insert(0, str(_HUAWEI_SDK))

_VCS_AVAILABLE = False
try:
    from huaweicloud_sis.bean.sis_config import SisConfig
    from huaweicloud_sis.client.vcs_stream_client import VcsStreamClient
    from huaweicloud_sis.bean.vcs_stream_request import VcsStreamRequest
    from huaweicloud_sis.bean.callback import VcsStreamCallBack

    class _AudioCallback(VcsStreamCallBack):
        """VCS 流式合成回调：将音频数据写入文件。"""

        def __init__(self, output_path: str):
            super().__init__()
            self._f = open(output_path, "wb")

        def on_response(self, data):
            self._f.write(data)

        def on_complete(self):
            self._f.close()

    _VCS_AVAILABLE = True
except ImportError:
    print("[播报] 华为云 VCS SDK 未找到，语音播报不可用。")


# ============================================================
#  VCS 语音合成
# ============================================================

def _vcs_synthesize(
    text: str,
    output_path: str | None = None,
    voice_name: str | None = None,
    volume: int = 55,
    speed: int = 0,
    pitch: int = 3,
) -> str | None:
    """
    使用华为云 VCS 将文本合成为指定音色的语音。
    """
    if not _VCS_AVAILABLE:
        return None

    voice = voice_name or VOICE_NAME
    if not voice:
        print("[播报] VOICE_NAME 未设置，跳过语音合成。")
        return None

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp3", prefix="tts_")
        os.close(fd)

    try:
        config = SisConfig()
        config.set_connect_timeout(15)
        config.set_read_timeout(15)

        callback = _AudioCallback(output_path)
        client = VcsStreamClient(
            ak=HUAWEI_AK,
            sk=HUAWEI_SK,
            use_aksk=True,
            region=HUAWEI_REGION,
            project_id=HUAWEI_PROJECT_ID,
            callback=callback,
            config=config,
        )

        req = VcsStreamRequest()
        req.set_voice_name(voice)
        req.set_audio_format("mp3")
        req.set_text(text)
        req.set_volume(volume)
        req.set_speed(speed)
        req.set_pitch(pitch)

        client.synthesis(req)
        return output_path

    except Exception as exc:
        print(f"[播报] VCS 合成失败: {exc}")
        return None


# ============================================================
#  公开接口
# ============================================================

def broadcast_text(text: str) -> str | None:
    """
    将心理疏导文案合成为语音。

    返回: 音频路径或 None
    """
    if not TTS_ENABLED:
        return None
    if not _VCS_AVAILABLE:
        return None
    if not VOICE_NAME:
        print("[播报] 未配置 VOICE_NAME，跳过语音播报。")
        return None

    print(f"[播报] 合成语音 ({VOICE_NAME}): {text[:60]}...")
    return _vcs_synthesize(text)


def tts_available() -> bool:
    """检查语音播报是否可用。"""
    return TTS_ENABLED and _VCS_AVAILABLE and bool(VOICE_NAME)
