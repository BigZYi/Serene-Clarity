"""
app.py —— 人脸表情 → 心理状态 → 天气 → LLM 心理疏导 Gradio 网站

启动方式:
    python app.py

设计理念：静默监测 + 定时周期分析
    - 摄像头持续运行，每帧检测表情，实时累积情绪数据
    - 每隔 LLM_INTERVAL_SECONDS 秒，汇总周期内情绪，调用 LLM 生成疏导
    - 天气每天只请求一次
    - 用户也可随时点击按钮主动获取疏导
"""

from __future__ import annotations

import time

import numpy as np

try:
    import gradio as gr
except ImportError:
    raise ImportError("请先安装 gradio: pip install gradio")

from config import APP_TITLE, LLM_INTERVAL_SECONDS, PSYCH_COLORS, SNAPSHOT_INTERVAL, TTS_ENABLED, WEATHER_CITY
from core import (
    SessionState,
    classify_psychological_state,
    update_emotion_log,
    build_emotion_summary,
)
from llm import generate_counseling
from vision import detect_frame_emotion, render_overlay
from weather_service import get_weather_context

# 启动时预加载天气
_INIT_WEATHER = ""
try:
    _INIT_WEATHER = get_weather_context(WEATHER_CITY if WEATHER_CITY else None) or ""
except Exception:
    pass

# 语音播报（可选）
_tts_available = False
if TTS_ENABLED:
    try:
        import broadcast
        _tts_available = broadcast.tts_available()
    except Exception:
        pass


# ============================================================
#  CSS 样式
# ============================================================

CSS = """
/* ── 调色板 ── */
:root {
  --bg-deep: #1a0f08;
  --bg-mid: #2d1a0e;
  --card-bg: rgba(50, 30, 20, 0.85);
  --text-main: #f5e6d3;
  --text-muted: #b8a99a;
  --text-dim: #8b7355;
  --orange: #f97316;
  --orange-light: #fb923c;
  --orange-glow: #fbbf24;
  --border-subtle: rgba(255,255,255,0.06);
  --radius: 16px;
}

/* ── 全局 ── */
body, .gradio-container {
  background: linear-gradient(160deg, var(--bg-deep), var(--bg-mid) 40%, #1a1215);
  color: var(--text-main);
  font-family: "Segoe UI", system-ui, sans-serif;
}

/* ── 页头 ── */
#app-header { text-align: center; padding: 28px 20px 10px; }
#app-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 6px; color: var(--text-main); }
#app-header h1 span {
  color: #fbbf24;
}
#app-header p { color: var(--text-muted); font-size: 1rem; margin: 0; }

/* ── 卡片 ── */
.info-card {
  background: var(--card-bg);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius); padding: 20px; margin-bottom: 12px;
  text-align: center;
}
.info-card .label { color: var(--text-muted); font-size: 0.7rem; margin-bottom: 4px; }
.info-card .badge {
  display: inline-block; padding: 6px 16px; border-radius: 999px;
  font-size: 1.2rem; font-weight: 700;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
}
.info-card .hint { color: var(--text-dim); font-size: 0.7rem; margin-top: 6px; }

.weather-card {
  background: rgba(251,146,60,0.06); border: 1px solid rgba(251,146,60,0.18);
  border-radius: 14px; padding: 14px 18px; font-size: 0.9rem; color: #fed7aa;
}

.counsel-card {
  border-radius: var(--radius); padding: 20px; line-height: 1.8; font-size: 1.05rem;
  color: #fef3c7; border: 1px solid rgba(251,146,60,0.25);
  background: linear-gradient(135deg, rgba(251,146,60,0.08), rgba(245,158,11,0.04));
}
.counsel-card .label {
  font-size: 0.75rem; color: var(--orange-light); margin-bottom: 6px;
  text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── 按钮 ── */
button, .gr-button {
  background: rgba(251,146,60,0.14) !important;
  border: none !important;
  color: #fed7aa !important;
  border-radius: 10px !important;
}
button:hover, .gr-button:hover { background: rgba(251,146,60,0.24) !important; }
.primary, .gr-button-primary {
  background: var(--orange) !important;
  color: #fff !important;
}
.primary:hover, .gr-button-primary:hover { background: var(--orange-light) !important; }

/* ── 折叠面板 ── */
.gr-accordion { border: 1px solid var(--border-subtle) !important; border-radius: var(--radius) !important; }
"""



# ============================================================
#  定时周期分析
# ============================================================

AUDIO_MIN_INTERVAL = 10.0  # 两次音频之间至少间隔 10 秒（防打断播放）


def _should_update_audio(state: SessionState) -> bool:
    now = time.time()
    return (now - state.last_audio_set_time) >= AUDIO_MIN_INTERVAL


def _audio_out(state: SessionState, new_path: str | None) -> str | None:
    """仅在音频路径变化时返回新值，防 Gradio 重复播报。"""
    path = new_path or ""
    if path == state._last_audio_sent:
        return state._last_audio_sent or None  # 不变 → 返回相同值不触发刷新
    state._last_audio_sent = path
    return new_path


def _analyze_and_counsel(state: SessionState) -> tuple[str, str, str, str | None]:
    """
    汇总周期内的情绪数据，调用 LLM 生成疏导。
    返回 (counsel_html, weather_html, psych_label, audio_path)
    """
    # 获取天气（当天只请求一次，weather_service 已处理缓存）
    weather_ctx = ""
    try:
        from weather_service import get_weather_context as _wctx
        weather_ctx = _wctx(WEATHER_CITY if WEATHER_CITY else None) or ""
    except Exception:
        pass

    # 周期内主导情绪
    if state.emotion_log:
        from collections import Counter
        dominant = Counter(state.emotion_log).most_common(1)[0][0]
    else:
        dominant = "中性"

    psych = classify_psychological_state(dominant)
    psych_label = psych["label"]
    psych_strategy = psych["strategy"]
    state.psych_label = psych_label
    state.psych_strategy = psych_strategy

    emotion_summary = build_emotion_summary(state)
    counseling = generate_counseling(
        psych_label=psych_label,
        psych_strategy=psych_strategy,
        emotion=dominant,
        emotion_summary=emotion_summary,
        user_text="",
    )

    state.last_counsel_time = time.time()
    state.last_counseling = counseling
    state.last_counseling_time_str = time.strftime("%H:%M:%S")
    state.counsel_count += 1

    # ── 语音播报 ──
    audio_path = None
    if TTS_ENABLED and _tts_available:
        try:
            audio_path = broadcast.broadcast_text(counseling)
        except Exception:
            pass

    weather_html = (
        f"<div class='weather-card'>🌤 {weather_ctx}</div>"
        if weather_ctx
        else "<div class='weather-card' style='color:#b8a99a;'>天气数据暂不可用</div>"
    )
    counsel_html = (
        f"<div class='counsel-card'>"
        f"  <div class='label'>🕐 {state.last_counseling_time_str}</div>"
        f"  <div>{counseling}</div>"
        f"</div>"
    )

    return counsel_html, weather_html, psych_label, audio_path


# ============================================================
#  辅助 HTML 构建
# ============================================================

def _emotion_html(emotion: str, has_face: bool) -> str:
    face_status = "✅ 检测到人脸" if has_face else "⚠️ 未检测到人脸"
    return (
        f"<div class='info-card'>"
        f"  <div class='label'>🎭 人脸表情</div>"
        f"  <span class='badge'>{emotion}</span>"
        f"  <div class='hint'>{face_status}</div>"
        f"</div>"
    )


def _psych_html(psych_label: str, psych_strategy: str, next_in: float) -> str:
    color = PSYCH_COLORS.get(psych_label, "#b8a99a")
    return (
        f"<div class='info-card'>"
        f"  <div class='label'>🧠 心理状态</div>"
        f"  <span class='badge' style='background:{color}22;border:1px solid {color}44;color:{color};'>{psych_label}</span>"
        f"  <div class='hint'>{psych_strategy}</div>"
        f"</div>"
    )


# ============================================================
#  构建 Gradio 界面
# ============================================================

def build_app():
    state = SessionState()

    with gr.Blocks(title=APP_TITLE) as demo:
        # ── 页头 ──
        gr.HTML(
            """
            <div id="app-header">
              <h1>☀️ <span>心晴</span></h1>
              <p>静默守护</p>
            </div>
            """
        )

        # ── 主体卡片 ──
        with gr.Column():
            with gr.Row():
                emotion_display = gr.HTML(
                    value="<div class='info-card' style='text-align:center;color:#b8a99a;'>等待摄像头...</div>"
                )
                psych_display = gr.HTML(
                    value="<div class='info-card' style='text-align:center;color:#b8a99a;'>等待分析...</div>"
                )

            weather_display = gr.HTML(
                value=f"<div class='weather-card'>{'🌤 ' + _INIT_WEATHER if _INIT_WEATHER else '正在获取天气...'}</div>"
            )

            counsel_display = gr.HTML(
                value="<div class='counsel-card' style='color:#b8a99a;'>打开摄像头后将自动监测你的状态，在需要时给予疏导。</div>"
            )

            with gr.Row():
                user_input = gr.Textbox(
                    label="💬 此刻想说的话",
                    placeholder="今天感觉怎么样？……",
                    lines=2,
                    scale=3,
                )
                counsel_btn = gr.Button("💡 获取疏导", variant="primary", size="lg", scale=1)
                pause_btn = gr.Button("⏸ 暂停守护", size="lg")
                reset_btn = gr.Button("🔄 重置", size="lg")

            processed_frame = gr.Image(
                label="实时画面",
                type="numpy",
                visible=False,
            )

            tts_audio = gr.Audio(
                label="🔊 语音播报",
                type="filepath",
                autoplay=True,
                visible=TTS_ENABLED,
            )

        # ── 摄像头（底部折叠）──
        with gr.Accordion("📷 摄像头（仅本机分析 · 不录制不上传不存储）", open=False):
            camera_input = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
            )

        # ── 说明 ──
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(
                """
                ### 🤫 静默监测 + 定时周期分析

                无需任何操作——打开摄像头后，系统会自动：
                1. 每帧检测你的面部表情，累积情绪数据
                2. **每隔 N 秒**，汇总周期内所有情绪，调用 LLM 生成心理疏导
                3. 天气每天只请求一次（自动缓存）
                4. 实时画面显示当前表情 + 心理状态 + 距离下次疏导的倒计时

                ### 🔊 语音播报

                心理疏导生成后会自动语音播报（需在 `config.py` 中开启 `TTS_ENABLED` 并配置好音色）。

                ### ⏱ 周期配置

                修改 `config.py` 中的 `LLM_INTERVAL_SECONDS` 来调整分析周期（默认 30 秒）。

                ### 💡 主动获取

                输入想说的话后点击「获取疏导」，随时触发心理疏导，不受周期限制。

                ### 🛠 技术栈

                | 模块 | 技术 |
                |------|------|
                | 人脸检测 | MediaPipe Face Landmarker（478 点 + 52 blendshapes） |
                | 表情识别 | 几何特征 + blendshape 系数综合判断 |
                | 天气服务 | uapis.cn 实时天气 API（每天一次） |
                | 心理疏导 | DeepSeek LLM / 本地规则降级 |
                | 界面 | Gradio 6.x |
                """
            )

        # ============================================================
        #  回调：stream —— 仅更新实时画面 + 静默累积情绪
        # ============================================================

        def on_stream(frame):
            """
            拍立得模式：每帧到达，但仅每隔 SNAPSHOT_INTERVAL 秒分析一次。
            暂停时跳过所有分析。
            """
            nonlocal state
            if state is None:
                state = SessionState()

            if frame is None:
                return render_overlay(None, "中性", "等待中")

            # 暂停状态：不做任何分析
            if state.paused:
                return render_overlay(None, "中性", "已暂停")

            # 拍立得节流：不到间隔就跳过
            now = time.time()
            if state.last_snapshot_time > 0 and (now - state.last_snapshot_time) < SNAPSHOT_INTERVAL:
                return render_overlay(frame, state.last_emotion, state.psych_label)

            state.last_snapshot_time = now

            face_info = detect_frame_emotion(frame)
            emotion = face_info["emotion"]
            state = update_emotion_log(state, emotion)

            psych = classify_psychological_state(emotion)
            state.psych_label = psych["label"]
            state.psych_strategy = psych["strategy"]
            state.last_emotion = emotion
            state.last_has_face = face_info["has_face"]
            state.last_frame = frame

            return render_overlay(frame, emotion, psych["label"])

        # ============================================================
        #  回调：timer —— 每秒更新一次 HTML 显示 + 周期触发 LLM
        # ============================================================

        def on_timer():
            """每秒触发：读取 state，更新所有 HTML 卡片。到周期时调用 LLM。暂停时跳过 LLM。"""
            nonlocal state
            if state is None:
                state = SessionState()

            now = time.time()
            emotion = getattr(state, "last_emotion", "中性")
            has_face = getattr(state, "last_has_face", False)
            psych_label = state.psych_label
            psych_strategy = state.psych_strategy

            # ── 天气获取（一次导入，全 tick 共用）──
            try:
                from weather_service import get_weather_context as _wctx_tick
                _weather_now = _wctx_tick(WEATHER_CITY if WEATHER_CITY else None) or ""
            except Exception:
                _weather_now = ""
            weather_html = (
                f"<div class='weather-card'>🌤 {_weather_now}</div>"
                if _weather_now
                else "<div class='weather-card' style='color:#b8a99a;'>天气数据暂不可用</div>"
            )

            # 暂停时显示暂停状态，不触发 LLM
            if state.paused:
                audio_path = state.last_audio_path
                counsel_html = (
                    "<div class='counsel-card' style='color:#b8a99a;'>"
                    "⏸ 守护已暂停。点击「恢复守护」继续。"
                    "</div>"
                )
                emotion_html = _emotion_html(emotion, has_face)
                psych_html = _psych_html(psych_label, psych_strategy, 0)
                return emotion_html, psych_html, weather_html, counsel_html, audio_path

            # 周期检查
            elapsed = now - state.last_counsel_time if state.last_counsel_time > 0 else LLM_INTERVAL_SECONDS + 1
            next_in = max(0, LLM_INTERVAL_SECONDS - elapsed)

            if elapsed >= LLM_INTERVAL_SECONDS:
                # ── LLM 思考过渡 ──
                if not state.llm_thinking:
                    state.llm_thinking = True
                    audio_path = state.last_audio_path
                    # 保留上一次的文案，仅在角落提示「正在想…」
                    if state.last_counseling:
                        counsel_html = (
                            f"<div class='counsel-card'>"
                            f"  <div class='label'>🕐 {state.last_counseling_time_str} · 💭 正在分析你的状态…</div>"
                            f"  <div>{state.last_counseling}</div>"
                            f"</div>"
                        )
                    else:
                        counsel_html = (
                            "<div class='counsel-card' style='text-align:center;'>"
                            "  <div style='font-size:1.2rem;'>💭 正在分析…</div>"
                            "</div>"
                        )
                    emotion_html = _emotion_html(emotion, has_face)
                    psych_html = _psych_html(psych_label, psych_strategy, next_in)
                    return emotion_html, psych_html, weather_html, counsel_html, _audio_out(state, audio_path)

                # 下一个 tick：真正调用 LLM
                state.llm_thinking = False
                counsel_html, weather_html, _, audio_path = _analyze_and_counsel(state)
                if audio_path is not None and _should_update_audio(state):
                    state.last_audio_path = audio_path
                    state.last_audio_set_time = time.time()
            else:
                audio_path = state.last_audio_path  # 保持上一次的音频，不打断播放
                if state.last_counseling:
                    counsel_html = (
                        f"<div class='counsel-card'>"
                        f"  <div class='label'>🕐 {state.last_counseling_time_str}</div>"
                        f"  <div>{state.last_counseling}</div>"
                        f"</div>"
                    )
                else:
                    counsel_html = (
                        "<div class='counsel-card' style='color:#b8a99a;'>"
                        "监测中… 每隔一段时间会自动为你分析情绪并生成疏导。"
                        "</div>"
                    )

            emotion_html = _emotion_html(emotion, has_face)
            psych_html = _psych_html(psych_label, psych_strategy, next_in)

            return emotion_html, psych_html, weather_html, counsel_html, _audio_out(state, audio_path)

        # ============================================================
        #  回调：主动获取疏导
        # ============================================================

        def on_counsel(user_text: str):
            """用户输入文字 + 点击按钮获取心理疏导。"""
            nonlocal state
            if state is None:
                state = SessionState()

            emotion = getattr(state, "last_emotion", "中性")
            psych_label = state.psych_label
            psych_strategy = state.psych_strategy

            weather_ctx = ""
            try:
                from weather_service import get_weather_context as _wctx
                weather_ctx = _wctx(WEATHER_CITY if WEATHER_CITY else None) or ""
            except Exception:
                pass

            emotion_summary = build_emotion_summary(state)
            counseling = generate_counseling(
                psych_label=psych_label,
                psych_strategy=psych_strategy,
                emotion=emotion,
                emotion_summary=emotion_summary,
                user_text=user_text or "",
            )

            state.last_counsel_time = time.time()
            state.last_counseling = counseling
            state.last_counseling_time_str = time.strftime("%H:%M:%S")
            state.counsel_count += 1
            state.llm_thinking = False  # 取消定时器的待触发状态，避免覆盖

            weather_html = (
                f"<div class='weather-card'>🌤 {weather_ctx}</div>"
                if weather_ctx
                else "<div class='weather-card' style='color:#b8a99a;'>天气数据暂不可用</div>"
            )
            counsel_html = (
                f"<div class='counsel-card'>"
                f"  <div class='label'>� {state.last_counseling_time_str}</div>"
                f"  <div>{counseling}</div>"
                f"</div>"
            )

            # ── 语音播报 ──
            audio_path = None
            if TTS_ENABLED and _tts_available:
                try:
                    audio_path = broadcast.broadcast_text(counseling)
                    if audio_path is not None and _should_update_audio(state):
                        state.last_audio_path = audio_path
                        state.last_audio_set_time = time.time()
                except Exception:
                    pass

            return weather_html, counsel_html, _audio_out(state, audio_path or state.last_audio_path)

        # ============================================================
        #  回调：暂停 / 恢复
        # ============================================================

        def on_pause():
            nonlocal state
            if state is None:
                state = SessionState()
            state.paused = not state.paused
            label = "▶ 恢复守护" if state.paused else "⏸ 暂停守护"
            return gr.update(value=label)

        # ============================================================
        #  回调：重置
        # ============================================================

        def on_reset():
            nonlocal state
            state = SessionState()
            return (
                "<div class='info-card' style='text-align:center;color:#b8a99a;'>等待摄像头...</div>",
                "<div class='info-card' style='text-align:center;color:#b8a99a;'>等待分析...</div>",
                "<div class='weather-card'>正在获取天气...</div>",
                "<div class='counsel-card' style='color:#b8a99a;'>已重置。打开摄像头后将自动监测。</div>",
                None,
                "⏸ 暂停守护",
            )

        # ── 定时器：每秒刷新 HTML 显示 ──
        timer = gr.Timer(value=1.0)

        # ── 事件绑定 ──

        # Stream：只更新实时画面（高频，不牵连 HTML）
        camera_input.stream(
            fn=on_stream,
            inputs=[camera_input],
            outputs=[processed_frame],
        )

        # Timer：每秒更新 HTML 卡片 + 周期触发 LLM
        html_outputs = [emotion_display, psych_display, weather_display, counsel_display]
        timer.tick(fn=on_timer, inputs=[], outputs=html_outputs + [tts_audio])

        # 主动获取疏导
        counsel_btn.click(
            fn=on_counsel,
            inputs=[user_input],
            outputs=[weather_display, counsel_display, tts_audio],
        )

        # 重置
        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=html_outputs + [tts_audio, pause_btn],
        )

        # 暂停 / 恢复
        pause_btn.click(
            fn=on_pause,
            inputs=[],
            outputs=[pause_btn],
        )

    return demo


# ============================================================
#  启动入口
# ============================================================

def main():
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
