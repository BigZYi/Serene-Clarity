"""
天气服务 —— 封装 weather.py，每天只调用一次 API。
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from weather import fetch_weather

_cache: dict[str, Any] = {}
_cache_date: str = ""  # 缓存日期，格式 YYYY-MM-DD


def _is_today(d: str) -> bool:
    return d == str(date.today())


def get_weather_data(city: str | None = None) -> dict:
    """获取天气数据，当天内使用缓存。"""
    global _cache, _cache_date

    # 缓存为空 或 不是今天的 → 重新请求
    if not _cache or not _is_today(_cache_date):
        try:
            _cache = fetch_weather(city)
            _cache_date = str(date.today())
        except Exception:
            pass  # 请求失败保留旧缓存（即使是昨天的也比没有好）

    return _cache


def refresh_weather(city: str | None = None) -> dict:
    """强制刷新天气（忽略缓存）。"""
    global _cache, _cache_date
    try:
        _cache = fetch_weather(city)
        _cache_date = str(date.today())
    except Exception:
        pass
    return _cache


def get_weather_context(city: str | None = None) -> str:
    """返回适合嵌入 LLM 提示词的天气上下文文本。"""
    data = get_weather_data(city)
    if not data:
        return "天气数据暂不可用。"

    weather = data.get("weather", "")
    temp = data.get("temperature", "")
    humidity = data.get("humidity", "")
    wind_dir = data.get("wind_direction", "")
    wind_power = data.get("wind_power", "")
    city_name = data.get("city", "")
    province = data.get("province", "")

    parts = [f"当前天气：{province}{city_name}，{weather}，{temp}°C"]
    if humidity:
        parts.append(f"，湿度{humidity}%")
    if wind_dir and wind_power:
        parts.append(f"，{wind_dir}{wind_power}")
    return "".join(parts) + "。"
