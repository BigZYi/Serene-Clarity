"""
weather.py —— 通过 uapis.cn 天气 API 获取实时天气。

API: https://uapis.cn/api/v1/misc/weather
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Any

API_URL = "https://uapis.cn/api/v1/misc/weather"
API_KEY = "uapi-xbe-ny_ncZ8PF4iFXrhRoJ-kS59eL7fi4EdVkfnq"


def fetch_weather(city: str | None = None) -> dict[str, Any]:
    """调用天气 API 并返回 JSON 数据。"""
    params: dict[str, str] = {}
    if city:
        params["city"] = city

    url = API_URL
    if params:
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("User-Agent", "EmotionWeatherApp/1.0")

    with urllib.request.urlopen(req, timeout=10) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    return data


def format_weather(data: dict[str, Any]) -> str:
    """将天气数据格式化为可读文本。"""
    lines: list[str] = []
    province = data.get("province", "")
    city = data.get("city", "")
    location = " · ".join(p for p in [province, city] if p)
    lines.append(f"📍 {location}")

    weather = data.get("weather", "未知")
    temp = data.get("temperature", "N/A")
    lines.append(f"🌤 {weather}  |  🌡 {temp}°C")

    wind_dir = data.get("wind_direction", "")
    wind_power = data.get("wind_power", "")
    humidity = data.get("humidity", "")
    lines.append(f"💨 {wind_dir} {wind_power}  |  💧 湿度 {humidity}%")

    return "\n".join(lines)
