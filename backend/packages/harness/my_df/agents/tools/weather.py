"""天气查询工具：基于 wttr.in 免费接口（无需 API key）。"""

import logging

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

_WTTR_IN_URL = "https://wttr.in/"

# wttr.in 的 lang=zh 参数当前不生效，手动维护英文天气描述 → 中文映射
_WEATHER_DESC_ZH: dict[str, str] = {
    "sunny": "晴",
    "clear": "晴",
    "clear sky": "晴空",
    "partly cloudy": "多云",
    "cloudy": "阴",
    "overcast": "阴天",
    "mist": "薄雾",
    "fog": "雾",
    "freezing fog": "冻雾",
    "light drizzle": "毛毛雨",
    "drizzle": "毛毛雨",
    "light rain": "小雨",
    "light rain shower": "小阵雨",
    "patchy rain nearby": "局部有雨",
    "patchy rain possible": "可能有阵雨",
    "moderate rain": "中雨",
    "moderate rain at times": "时有大雨",
    "heavy rain": "大雨",
    "heavy rain shower": "强阵雨",
    "torrential rain shower": "特大暴雨",
    "thundery outbreaks possible": "可能雷暴",
    "thunderstorm": "雷阵雨",
    "light snow": "小雪",
    "patchy snow nearby": "局部有雪",
    "moderate snow": "中雪",
    "heavy snow": "大雪",
    "sleet": "雨夹雪",
    "hail": "冰雹",
    "blowing snow": "风雪",
    "light freezing rain": "冻雨",
}


def _translate_desc(desc: str) -> str:
    """英文天气描述 → 中文；无法匹配时原样返回。"""
    if not desc:
        return ""
    key = desc.strip().lower()
    if key in _WEATHER_DESC_ZH:
        return _WEATHER_DESC_ZH[key]
    for en, zh in _WEATHER_DESC_ZH.items():
        if en in key:
            return zh
    return desc


def _day_description(hourly: list[dict] | None) -> str:
    """从 hourly 预报中提取当天描述（lang_zh 优先，回退英文再翻译）。"""
    if not hourly:
        return ""
    first = hourly[0]
    for field in ("lang_zh", "weatherDesc"):
        entries = first.get(field) or []
        if entries and entries[0].get("value"):
            return _translate_desc(entries[0]["value"])
    return ""


def _format_weather(data: dict, city: str, date: str) -> str:
    """把 wttr.in JSON 格式化为可读文本。"""
    lines = [f"{city} 天气" + (f"（{date}）" if date else "")]
    current = (data.get("current_condition") or [{}])[0]
    if current:
        desc = _translate_desc((current.get("weatherDesc") or [{}])[0].get("value", ""))
        lines.append(
            f"当前：{desc}，{current.get('temp_C', '?')}°C"
            f"（体感 {current.get('FeelsLikeC', '?')}°C），"
            f"湿度 {current.get('humidity', '?')}%，"
            f"风速 {current.get('windspeedKmph', '?')} km/h"
        )

    forecast = data.get("weather") or []
    for day in forecast[:3]:
        desc = _day_description(day.get("hourly"))
        lines.append(
            f"{day.get('date', '?')}：{desc or '未知'}，"
            f"{day.get('mintempC', '?')}~{day.get('maxtempC', '?')}°C"
        )
    return "\n".join(lines)


@tool
def search_weather(city: str, date: str = "") -> str:
    """查询指定城市在指定日期的天气。

    参数：
        city: 城市名，如“武汉”“北京”（支持中文）。
        date: 日期（可选），格式 YYYY-MM-DD；为空时返回当前及未来两天预报。

    返回：
        人类可读的天气信息文本；查询失败时返回错误说明。
    """
    # 注意：wttr.in 的 JSON 格式参数是 j1（format=j 当前返回字面 "j"）；
    # 日期查询走 URL path（?date= 参数当前返回 ERR004 not implemented）
    url = f"{_WTTR_IN_URL}{city}"
    if date:
        url = f"{url}/{date}"
    try:
        resp = httpx.get(url, params={"format": "j1"}, timeout=15)
        if resp.status_code == 500 or "Unknown location" in resp.text:
            return f"未找到城市“{city}”的天气信息，请检查城市名。"
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("天气查询失败 city=%s date=%s: %s", city, date, e)
        return f"天气查询失败：{e}"
    return _format_weather(data, city, date)
