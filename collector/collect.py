#!/usr/bin/env python3
"""Сборщик погодных и морских данных для береговой рыбалки на леврека
(Алания — Махмутлар — Газипаша).

Тянет Open-Meteo (Forecast + Marine), считает производные значения
(состояние устьев, окна ловли, оценку) и пишет:
  windows   — четыре окна в сутки: рассвет, день, закат, ночь (для дня
              и ночи — лучший 3-часовой отрезок периода);
  timeline  — почасовая оценка по каждой прибрежной точке на 7 дней.
Файлы:
  data/latest.json          — компактный результат для чтения моделью
  data/summary.md           — сводка по-русски
  data/raw/latest_raw.json  — сырые ответы API (перезаписывается)
  data/history/YYYY-MM-DD.json — снимок latest.json на день

Только стандартная библиотека Python 3.12.
Данные: Open-Meteo.com, CC BY 4.0.
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Istanbul")
except Exception:  # нет tzdata — Турция с 2016 года живёт в UTC+3 без перехода
    TZ = timezone(timedelta(hours=3), "Europe/Istanbul")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "points.json")
DATA = os.path.join(ROOT, "data")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
META_URL = "https://api.open-meteo.com/data/{model}/static/meta.json"
MARINE_META_URL = "https://marine-api.open-meteo.com/data/{model}/static/meta.json"

# past_days=3, а не 2: правилам устьев нужен пик за полные 72 ч,
# а при past_days=2 история начинается с 00:00 позавчера (48–71 ч назад).
FORECAST_PAST_DAYS = 3
FORECAST_DAYS = 7
MARINE_PAST_DAYS = 2           # волна за прошлые 48 ч — для графика
TIMELINE_PAST_HOURS = 48       # почасовая лента начинается за 48 ч до текущего часа

HOURLY = ["precipitation", "rain", "showers", "weather_code", "cloud_cover",
          "pressure_msl", "wind_speed_10m", "wind_direction_10m",
          "wind_gusts_10m", "cape", "temperature_2m"]
DAILY = ["sunrise", "sunset", "precipitation_sum"]
MARINE_HOURLY = ["wave_height", "wave_direction", "wave_period",
                 "wind_wave_height", "swell_wave_height", "swell_wave_direction",
                 "swell_wave_period", "sea_surface_temperature"]

# Модели, чьи запуски показываем в source_model_runs. best_match смешивает
# модели и сам не сообщает, какие именно, поэтому это справочно.
FORECAST_MODELS = ["ecmwf_ifs025", "dwd_icon", "ncep_gfs025"]
MARINE_MODELS = ["meteofrance_wave"]

COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print("WARN:", msg, file=sys.stderr)


# ---------------------------------------------------------------- сеть

def http_json(url, params=None, attempts=5, timeout=60):
    """GET с ретраями и экспоненциальным backoff (2, 4, 8, 16 с)."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params, safe=",")
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "velvet-turnip/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {body}"
            if 400 <= e.code < 500 and e.code != 429:
                break  # ошибка запроса, повтор не поможет
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = repr(e)
        if i < attempts - 1:
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"{url[:120]}... -> {last}")


# ---------------------------------------------------------------- утилиты

def parse_local(s):
    """'2026-10-26T07:09' (время Europe/Istanbul) -> aware datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=TZ)


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M")


def rnd(x, n=1):
    return None if x is None else round(x, n)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compass(deg):
    if deg is None:
        return None
    return COMPASS16[int((deg % 360) / 22.5 + 0.5) % 16]


def vector_mean_dir(speeds, dirs):
    """Средний румб ветра: векторное среднее, взвешенное скоростью."""
    u = v = 0.0
    n = 0
    for s, d in zip(speeds, dirs):
        if s is None or d is None:
            continue
        u += s * math.sin(math.radians(d))
        v += s * math.cos(math.radians(d))
        n += 1
    if n == 0 or (u == 0 and v == 0):
        return None
    return math.degrees(math.atan2(u, v)) % 360


class Series:
    """Почасовой ряд одной точки: times[i] -> значения по параметрам."""

    def __init__(self, hourly):
        self.times = [parse_local(t) for t in hourly["time"]]
        self.data = {k: v for k, v in hourly.items() if k != "time"}
        self.index = {t: i for i, t in enumerate(self.times)}

    def get(self, var, t):
        i = self.index.get(t)
        if i is None or var not in self.data:
            return None
        return self.data[var][i]

    def between(self, var, start, end, incl_start=True):
        """Значения var для часов start <= t <= end (или start < t)."""
        out = []
        for i, t in enumerate(self.times):
            if (t >= start if incl_start else t > start) and t <= end:
                out.append(self.data.get(var, [None] * len(self.times))[i])
        return out


# ---------------------------------------------------------------- Луна

def moon_phase(dt):
    """Фаза Луны по среднему синодическому месяцу от новолуния
    2000-01-06 18:14 UTC (JD 2451550.26). Точность ~±0.5 сут — для
    рыбалки достаточно. Освещённость = (1 - cos(2π·возраст/P)) / 2."""
    P = 29.530588853
    jd = dt.astimezone(timezone.utc).timestamp() / 86400 + 2440587.5
    age = (jd - 2451550.26) % P
    illum = (1 - math.cos(2 * math.pi * age / P)) / 2 * 100
    names = [(1.85, "новолуние"), (5.54, "растущий серп"), (9.23, "первая четверть"),
             (12.92, "растущая луна"), (16.61, "полнолуние"), (20.30, "убывающая луна"),
             (23.99, "последняя четверть"), (27.68, "убывающий серп"), (P + 1, "новолуние")]
    name = next(n for lim, n in names if age < lim)
    to_full = (P / 2 - age) % P
    to_new = (P - age) % P
    return {
        "illumination_pct": round(illum),
        "phase": name,
        "age_days": round(age, 1),
        "waxing": age < P / 2,
        "next_full_moon": (dt + timedelta(days=to_full)).strftime("%Y-%m-%d"),
        "next_new_moon": (dt + timedelta(days=to_new)).strftime("%Y-%m-%d"),
    }


def moon_altitude(dt, lat, lon):
    """Высота Луны над горизонтом, градусы. Упрощённая модель (как в
    suncalc): точность ~1°, для «Луна над горизонтом или нет» хватает."""
    rad = math.pi / 180
    d = dt.astimezone(timezone.utc).timestamp() / 86400 + 2440587.5 - 2451545.0
    L = rad * (218.316 + 13.176396 * d)
    M = rad * (134.963 + 13.064993 * d)
    F = rad * (93.272 + 13.229350 * d)
    lon_ecl = L + rad * 6.289 * math.sin(M)
    lat_ecl = rad * 5.128 * math.sin(F)
    e = rad * 23.4397
    ra = math.atan2(math.sin(lon_ecl) * math.cos(e) - math.tan(lat_ecl) * math.sin(e), math.cos(lon_ecl))
    dec = math.asin(math.sin(lat_ecl) * math.cos(e) + math.cos(lat_ecl) * math.sin(e) * math.sin(lon_ecl))
    H = rad * (280.16 + 360.9856235 * d) + rad * lon - ra
    phi = rad * lat
    alt = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(H))
    return alt / rad


def moonlight(dt, lat, lon, cloud_pct):
    """Эффективный лунный свет 0..1: освещённость диска × доля чистого неба,
    0 если Луна под горизонтом."""
    if moon_altitude(dt, lat, lon) <= 0:
        return 0.0
    illum = moon_phase(dt)["illumination_pct"] / 100
    clear = 1 - (cloud_pct if cloud_pct is not None else 50) / 100
    return illum * clear


# ---------------------------------------------------------------- устья

def basin_precip(series_list):
    """Средние почасовые осадки по точкам водосбора (None игнорируются)."""
    base = series_list[0]
    out = []
    for i, t in enumerate(base.times):
        out.append(mean([s.get("precipitation", t) for s in series_list]))
    return base.times, out


def river_stats(times, precip, at):
    """Осадки в водосборе относительно момента at (час, округлённый вниз).
    Осадки за час T — это сумма за (T-1ч, T], поэтому "за 24 ч до at" =
    часы (at-24ч, at]."""
    def total(a, b):
        vals = [p for t, p in zip(times, precip) if a < t <= b]
        return sum(p for p in vals if p is not None), len(vals)

    p24, _ = total(at - timedelta(hours=24), at)
    p48, _ = total(at - timedelta(hours=48), at)
    f24, _ = total(at, at + timedelta(hours=24))
    f48, _ = total(at, at + timedelta(hours=48))
    window = [(t, p) for t, p in zip(times, precip)
              if at - timedelta(hours=72) < t <= at and p is not None]
    peak_t, peak_v = None, 0.0
    for t, p in window:
        # Пиком считаем только час с осадками >= 0,1 мм; при равенстве — более ранний.
        if p >= 0.1 and p > peak_v:
            peak_t, peak_v = t, p
    hours_since = None if peak_t is None else (at - peak_t).total_seconds() / 3600
    return {
        "past_24h_mm": round(p24, 1),
        "past_48h_mm": round(p48, 1),
        "next_24h_mm": round(f24, 1),
        "next_48h_mm": round(f48, 1),
        "peak_72h_time": fmt(peak_t) if peak_t else None,
        "peak_72h_mm_per_h": round(peak_v, 1) if peak_t else None,
        "hours_since_peak": rnd(hours_since, 0),
    }


def mouth_state(st):
    """Правила состояния устья (порядок важен):
      dry_or_low — за 48 ч < 10 мм;
      flood_now  — пик < 12 ч назад и за 24 ч >= 25 мм (мутный паводок);
      prime      — с пика 12–36 ч и за 48 ч >= 10 мм (вынос корма, лучшее);
      fading     — с пика 36–72 ч;
      rising     — дождь идёт/был < 12 ч назад, но меньше 25 мм за 24 ч
                   (правилами из задания не покрыт: «вода прибывает»)."""
    p24, p48, h = st["past_24h_mm"], st["past_48h_mm"], st["hours_since_peak"]
    if p48 < 10 or h is None:
        return "dry_or_low"
    if h < 12 and p24 >= 25:
        return "flood_now"
    if 12 <= h <= 36:
        return "prime"
    if 36 < h <= 72:
        return "fading"
    return "rising"


STATE_RU = {
    "dry_or_low": "сухо / низкая вода",
    "flood_now": "паводок сейчас",
    "prime": "лучшее время (prime)",
    "fading": "затухает",
    "rising": "вода прибывает",
}


# ---------------------------------------------------------------- оценка

OFFSHORE_SECTORS = {"N", "NNE", "NE"}   # ветер с суши для южного берега

# Поправки по времени суток (light: twilight / day / night).
# Сумерки — базовый случай, без поправок. Днём леврек у берега держится
# хуже: пасмурно или дождь возвращают день к уровню сумерек, но не выше;
# ясно и тихо — ещё хуже. Ночью решает освещённость: тёмная ночь лучше,
# яркая Луна при ясном небе хуже.
LIGHT_MODS = {
    "day_base": -1,                 # день сам по себе хуже сумерек
    "day_overcast_cloud": 70,       # облачность выше, % — +1 (компенсирует day_base)
    "day_clear_cloud": 40,          # ясно (облачность ниже, %) ...
    "day_clear_wave": 0.3,          # ... и волна ниже, м — ещё −1
    "night_dark": 0.2,              # эффективный лунный свет ниже — +1
    "night_bright": 0.7,            # выше — −1
}
LIGHT_RU = {"twilight": "сумерки", "day": "день", "night": "ночь"}


def score_window(w, nearby_prime, light="twilight"):
    """Прозрачная оценка окна 1..5. База 2 («обычно»), дальше поправки.
    Все пороги — здесь и в LIGHT_MODS, крутить можно прямо тут.
    light — время суток окна; для day/night добавляются поправки.
    Возвращает (score, reasons, flags)."""
    reasons = []
    flags = {"no_go": False, "dangerous_on_rocks": False}
    score = 2
    reasons.append("база 2")

    wave = w["wave_mean_m"]
    wave_max = w["wave_max_m"]
    wind = w["wind_mean_ms"]

    # Гроза: на берегу с удилищем опасно — сразу 1 и no_go.
    if w["thunder"]:
        flags["no_go"] = True
        if wave_max is not None and wave_max > 1.8:
            flags["dangerous_on_rocks"] = True
        return 1, ["гроза в окне или рядом (код 95–99 или CAPE > 1000) — не идти"], flags

    # Волна 0,4–1,2 м: прибой поднимает корм, леврек выходит к берегу.
    if wave is not None and 0.4 <= wave <= 1.2:
        score += 1
        reasons.append(f"волна {wave:.1f} м в рабочем диапазоне 0,4–1,2 м (+1)")

    # Нагонный ветер умеренной силы — гонит прибой и корм к берегу.
    if w["onshore"] and wind is not None and 3 <= wind <= 10:
        score += 1
        reasons.append(f"нагонный ветер {w['wind_dir']} {wind:.1f} м/с (+1)")

    # Сильный ветер с суши сглаживает прибой у берега.
    if w["wind_dir"] in OFFSHORE_SECTORS and wind is not None and wind > 4:
        score -= 1
        reasons.append(f"ветер с суши {w['wind_dir']} {wind:.1f} м/с (−1)")

    # Рядом устье в лучшей фазе после дождя.
    if nearby_prime:
        score += 1
        reasons.append("рядом устье в фазе prime: " + ", ".join(nearby_prime) + " (+1)")

    # Время суток.
    cloud = w.get("cloud_mean_pct")
    if light == "day":
        score += LIGHT_MODS["day_base"]
        reasons.append(f"день: у берега рыба держится хуже, чем в сумерки ({LIGHT_MODS['day_base']:+d})")
        if (cloud is not None and cloud > LIGHT_MODS["day_overcast_cloud"]) or (w.get("precip_mm") or 0) > 0.2:
            score += 1
            reasons.append(f"пасмурно или дождь (облачность {cloud if cloud is not None else '—'}%) — свет приглушён (+1)")
        if cloud is not None and cloud < LIGHT_MODS["day_clear_cloud"] and wave is not None and wave < LIGHT_MODS["day_clear_wave"]:
            score -= 1
            reasons.append(f"ясно ({cloud}%) и тихо ({wave:.2f} м) — прозрачная вода днём (−1)")
    elif light == "night":
        ml = w.get("moonlight")
        if ml is not None and ml < LIGHT_MODS["night_dark"]:
            score += 1
            reasons.append(f"тёмная ночь (лунный свет {ml:.2f}) (+1)")
        elif ml is not None and ml > LIGHT_MODS["night_bright"]:
            score -= 1
            reasons.append(f"яркая Луна при чистом небе (лунный свет {ml:.2f}) (−1)")

    # Штиль: без волны леврек осторожен — не выше 2.
    if wave is not None and wave < 0.2:
        if score > 2:
            reasons.append(f"волна {wave:.2f} м < 0,2 м — штиль, оценка не выше 2")
        else:
            reasons.append(f"волна {wave:.2f} м < 0,2 м — штиль")
        score = min(score, 2)

    if wave is None:
        reasons.append("нет данных по волне")

    if wave_max is not None and wave_max > 1.8:
        flags["dangerous_on_rocks"] = True
        reasons.append(f"волна до {wave_max:.1f} м — опасно на камнях")

    return max(1, min(5, score)), reasons, flags


# ---------------------------------------------------------------- окна

def window_metrics(name, kind, start, end, wx, sea, quiet=False):
    """Сводка погоды и моря для окна [start, end]. quiet — не писать
    предупреждения о пропусках (для почасовой ленты, иначе спам)."""
    sp = wx.between("wind_speed_10m", start, end)
    cl = wx.between("cloud_cover", start, end)
    dr = wx.between("wind_direction_10m", start, end)
    gu = wx.between("wind_gusts_10m", start, end)
    # Осадки за час T относятся к (T-1, T]: берём часы, пересекающие окно.
    pr = wx.between("precipitation", start, end + timedelta(hours=1), incl_start=False)
    codes = wx.between("weather_code", start - timedelta(hours=2), end + timedelta(hours=2))
    capes = wx.between("cape", start - timedelta(hours=2), end + timedelta(hours=2))
    wh = sea.between("wave_height", start, end) if sea else []
    wp = sea.between("wave_period", start, end) if sea else []

    t0 = start.replace(minute=0)
    p_now = wx.get("pressure_msl", t0)
    p_before = wx.get("pressure_msl", t0 - timedelta(hours=6))
    trend = None if p_now is None or p_before is None else p_now - p_before

    wind = mean(sp)
    d = vector_mean_dir(sp, dr)
    thunder = any(c is not None and 95 <= c <= 99 for c in codes) or \
        any(c is not None and c > 1000 for c in capes)

    missing = [k for k, v in (("ветер", sp), ("волна", wh)) if not v or all(x is None for x in v)]
    if missing and not quiet:
        warn(f"{name} {kind} {fmt(start)}: нет данных ({', '.join(missing)})")

    return {
        "wind_mean_ms": rnd(wind),
        "wind_dir_deg": rnd(d, 0),
        "wind_dir": compass(d),
        "gust_max_ms": rnd(max([g for g in gu if g is not None], default=None)),
        "onshore": d is not None and 90 <= d <= 270,
        "wave_mean_m": rnd(mean(wh), 2),
        "wave_max_m": rnd(max([x for x in wh if x is not None], default=None), 2),
        "wave_period_s": rnd(mean(wp)),
        "precip_mm": round(sum(p for p in pr if p is not None), 1),
        "thunder": thunder,
        "pressure_hpa": rnd(p_now),
        "pressure_trend_6h_hpa": rnd(trend),
        "cloud_mean_pct": None if mean(cl) is None else int(round(mean(cl))),
    }


# ---------------------------------------------------------------- main

def main():
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    now = datetime.now(TZ)
    now_hour = now.replace(minute=0, second=0, microsecond=0)

    # Точки прогноза: прибрежные + все точки водосборов, один запрос.
    fpoints = [("coastal", p["id"], p["lat"], p["lon"]) for p in cfg["coastal"]]
    for r in cfg["rivers"]:
        for c in r["catchment"]:
            fpoints.append(("catchment", c["id"], c["lat"], c["lon"]))
    mpoints = [("marine", p["id"], p["lat"], p["lon"]) for p in cfg["marine"]]

    common = {"timezone": "Europe/Istanbul", "wind_speed_unit": "ms"}
    try:
        fc = http_json(FORECAST_URL, dict(common,
            latitude=",".join(str(p[2]) for p in fpoints),
            longitude=",".join(str(p[3]) for p in fpoints),
            hourly=",".join(HOURLY), daily=",".join(DAILY),
            past_days=FORECAST_PAST_DAYS, forecast_days=FORECAST_DAYS))
    except Exception as e:
        print("FATAL: forecast API недоступен:", e, file=sys.stderr)
        return 2
    try:
        mc = http_json(MARINE_URL, dict(common,
            latitude=",".join(str(p[2]) for p in mpoints),
            longitude=",".join(str(p[3]) for p in mpoints),
            hourly=",".join(MARINE_HOURLY),
            past_days=MARINE_PAST_DAYS, forecast_days=FORECAST_DAYS))
    except Exception as e:
        print("FATAL: marine API недоступен:", e, file=sys.stderr)
        return 3
    if isinstance(fc, dict):
        fc = [fc]
    if isinstance(mc, dict):
        mc = [mc]
    if len(fc) != len(fpoints) or any("hourly" not in x for x in fc):
        print("FATAL: forecast вернул неполный ответ", file=sys.stderr)
        return 4
    if len(mc) != len(mpoints) or any("hourly" not in x for x in mc):
        print("FATAL: marine вернул неполный ответ", file=sys.stderr)
        return 5

    wx = {p[1]: Series(x["hourly"]) for p, x in zip(fpoints, fc)}
    sea = {p[1]: Series(x["hourly"]) for p, x in zip(mpoints, mc)}
    daily = {p[1]: x.get("daily") for p, x in zip(fpoints, fc)}

    # Критично: у морских точек волна должна быть хоть где-то.
    for pid, s in sea.items():
        if all(v is None for v in s.data.get("wave_height", [])):
            warn(f"{pid}: wave_height целиком null")
    if all(all(v is None for v in s.data.get("wave_height", [])) for s in sea.values()):
        print("FATAL: marine без волны во всех точках", file=sys.stderr)
        return 6

    # Пропуски по точкам прогноза в будущем горизонте -> warnings.
    for pid, s in wx.items():
        for var in ("precipitation", "wind_speed_10m", "pressure_msl"):
            vals = s.between(var, now_hour - timedelta(hours=72), now_hour + timedelta(hours=72))
            nn = sum(v is None for v in vals)
            if nn:
                warn(f"{pid}: {var} — {nn} пустых часов в диапазоне ±72 ч")

    # Модельные запуски (не критично).
    runs = {}
    for m, tmpl in [(m, META_URL) for m in FORECAST_MODELS] + [(m, MARINE_META_URL) for m in MARINE_MODELS]:
        try:
            meta = http_json(tmpl.format(model=m), attempts=2, timeout=20)
            t = meta.get("last_run_initialisation_time")
            runs[m] = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if t else None
        except Exception as e:
            warn(f"метаданные модели {m} недоступны: {e}")

    # ------------------------------------------------ устья
    rivers_out = []
    river_series = {}
    for r in cfg["rivers"]:
        ss = [wx[c["id"]] for c in r["catchment"]]
        times, pr = basin_precip(ss)
        river_series[r["id"]] = (times, pr)
        st = river_stats(times, pr, now_hour)
        item = {"id": r["id"], "name": r["name"], "mouth": {"lat": r["mouth"][0], "lon": r["mouth"][1]},
                "nearby_coastal": r["nearby_coastal"], **st}
        item["mouth_state"] = mouth_state(st)
        item["mouth_state_ru"] = STATE_RU[item["mouth_state"]]
        if r.get("dam"):
            # Отдельно ниже/выше плотины: ниже — сразу в устье, выше — через водохранилище.
            zones = {}
            for zone in ("below_dam", "above_dam"):
                zs = [wx[c["id"]] for c in r["catchment"] if c["zone"] == zone]
                if zs:
                    zt, zp = basin_precip(zs)
                    z = river_stats(zt, zp, now_hour)
                    zones[zone] = {k: z[k] for k in ("past_24h_mm", "past_48h_mm", "next_24h_mm")}
            item["zones"] = zones
            item["regulated"] = True
        if r.get("note"):
            item["note"] = r["note"]
        # Суточные суммы осадков в водосборе (по местной дате; час T относится
        # к дате часа T) — для графика «дождь в горах».
        daily_mm = {}
        for t, p_ in zip(times, pr):
            if p_ is not None:
                key = t.strftime("%Y-%m-%d")
                daily_mm[key] = daily_mm.get(key, 0.0) + p_
        item["daily_mm"] = [{"date": k, "mm": round(v, 1)} for k, v in sorted(daily_mm.items())]
        rivers_out.append(item)

    # ------------------------------------------------ окна ловли
    # Сутки делятся на четыре окна: рассвет и закат (фиксированные интервалы
    # вокруг восхода и захода), день и ночь (промежутки между ними; для них
    # берётся лучший 3-часовой отрезок). Плюс почасовая лента оценок.
    DAWN = (timedelta(hours=-1), timedelta(hours=1, minutes=30))
    DUSK = (timedelta(hours=-1, minutes=-30), timedelta(minutes=30))
    BLOCK = timedelta(hours=3)
    horizon = now + timedelta(days=7)

    def near_mouths(pid, at):
        """Состояние ближних устьев на момент at (с учётом прогноза осадков)."""
        near = {}
        for r in cfg["rivers"]:
            if pid in r["nearby_coastal"]:
                times, pr = river_series[r["id"]]
                near[r["name"]] = mouth_state(river_stats(times, pr, at))
        return near

    def mean_moonlight(p, start, end, s_wx):
        vals = []
        t = start.replace(minute=0)
        while t <= end:
            vals.append(moonlight(t, p["lat"], p["lon"], s_wx.get("cloud_cover", t)))
            t += timedelta(hours=1)
        return rnd(mean(vals), 2)

    def build_window(p, day, kind, light, start, end, s_wx, s_sea, quiet=False):
        w = {"point": p["id"], "point_name": p["name"], "date": day, "kind": kind,
             "light": light, "start": fmt(start), "end": fmt(end)}
        w.update(window_metrics(p["id"], kind, start, end, s_wx, s_sea, quiet=quiet))
        w["moonlight"] = mean_moonlight(p, start, end, s_wx)
        w["nearby_mouths"] = near_mouths(p["id"], start.replace(minute=0))
        prime = [n for n, s in w["nearby_mouths"].items() if s == "prime"]
        score, reasons, flags = score_window(w, prime, light)
        w["score"] = score
        w.update(flags)
        w["reasons"] = reasons
        w["moon_illumination_pct"] = moon_phase(start)["illumination_pct"]
        return w

    def best_block(p, day, kind, light, pstart, pend, s_wx, s_sea):
        """Лучший 3-часовой отрезок внутри [pstart, pend]: максимум оценки,
        при равенстве — без грозы, потом более ранний."""
        first = max(pstart, now_hour)
        t = first.replace(minute=0)
        if t < first:
            t += timedelta(hours=1)
        cands = []
        while t + BLOCK <= pend + timedelta(minutes=1):
            # 3 часа = три почасовых значения t, t+1, t+2.
            cands.append(build_window(p, day, kind, light, t, t + timedelta(hours=2), s_wx, s_sea, quiet=True))
            t += timedelta(hours=1)
        if not cands:
            return None
        best = max(cands, key=lambda w: (w["score"], not w["no_go"], -cands.index(w)))
        best["end"] = fmt(parse_local(best["start"]) + BLOCK)
        best["period_start"] = fmt(pstart)
        best["period_end"] = fmt(pend)
        best["period_thunder_hours"] = sum(1 for c in cands if c["thunder"])
        # Отрезок чистый, но в соседние часы этого же периода грозы: модель
        # ошибается со временем гроз на часы, поэтому −1.
        if not best["no_go"] and best["period_thunder_hours"] > 0:
            best["score"] = max(1, best["score"] - 1)
            best["reasons"].append("в другие часы этого периода грозы, время в модели неточное (−1)")
        best["period_precip_mm"] = round(sum(
            p_ for p_ in s_wx.between("precipitation", pstart, pend, incl_start=False) if p_ is not None), 1)
        return best

    windows = []
    sun = {}
    for p in cfg["coastal"]:
        d = daily[p["id"]]
        s_wx, s_sea = wx[p["id"]], sea.get(p["marine_id"])
        days = []
        for day, sr, ss in zip(d["time"], d["sunrise"], d["sunset"]):
            if not sr or not ss:
                warn(f"{p['id']} {day}: нет восхода/захода")
                continue
            days.append((day, parse_local(sr), parse_local(ss)))
        sun[p["id"]] = days
        for i, (day, sr, ss) in enumerate(days):
            dawn = (sr + DAWN[0], sr + DAWN[1])
            dusk = (ss + DUSK[0], ss + DUSK[1])
            for kind, (start, end) in (("рассвет", dawn), ("закат", dusk)):
                if end < now or start > horizon:
                    continue
                windows.append(build_window(p, day, kind, "twilight", start, end, s_wx, s_sea))
            # День: между рассветным и закатным окном.
            if dawn[1] < horizon and dusk[0] > now:
                w = best_block(p, day, "день", "day", dawn[1], dusk[0], s_wx, s_sea)
                if w:
                    windows.append(w)
            # Ночь: от конца заката до начала следующего рассвета; относится к дате вечера.
            if i + 1 < len(days):
                nstart, nend = dusk[1], days[i + 1][1] + DAWN[0]
                if nend > now and nstart < horizon:
                    w = best_block(p, day, "ночь", "night", nstart, nend, s_wx, s_sea)
                    if w:
                        windows.append(w)
    windows.sort(key=lambda w: (w["start"], w["point"]))

    # ------------------------------------------------ почасовая лента
    def light_at(t, days):
        for day, sr, ss in days:
            if sr + DAWN[0] <= t < sr + DAWN[1] or ss + DUSK[0] <= t < ss + DUSK[1]:
                return "twilight"
            if sr + DAWN[1] <= t < ss + DUSK[0]:
                return "day"
        return "night"

    timeline = {}
    for p in cfg["coastal"]:
        s_wx, s_sea = wx[p["id"]], sea.get(p["marine_id"])
        cols = {k: [] for k in ("time", "light", "score", "no_go", "thunder", "wind_ms", "wind_dir",
                                "wind_deg", "gust_ms", "onshore", "wave_m", "wave_period_s", "precip_mm",
                                "cloud_pct", "moonlight", "temp_c", "pressure_hpa")}
        # Лента начинается за 48 ч до текущего часа: для графиков нужен уже
        # выпавший дождь и ход давления. Оценки прошлых часов — справочно.
        t = now_hour - timedelta(hours=TIMELINE_PAST_HOURS)
        while t < horizon:
            light = light_at(t, sun[p["id"]])
            m = window_metrics(p["id"], "час", t, t, s_wx, s_sea, quiet=True)
            if m["wind_mean_ms"] is None and m["wave_mean_m"] is None:
                # За горизонтом прогноза данных нет — час не оцениваем.
                t += timedelta(hours=1)
                continue
            m["moonlight"] = rnd(moonlight(t, p["lat"], p["lon"], s_wx.get("cloud_cover", t)), 2)
            prime = [n for n, s in near_mouths(p["id"], t).items() if s == "prime"]
            sc, _, fl = score_window(m, prime, light)
            for k, v in (("time", fmt(t)), ("light", light), ("score", sc), ("no_go", fl["no_go"]),
                         ("thunder", m["thunder"]), ("wind_ms", m["wind_mean_ms"]), ("wind_dir", m["wind_dir"]),
                         ("wind_deg", m["wind_dir_deg"]), ("gust_ms", m["gust_max_ms"]),
                         ("onshore", m["onshore"]), ("wave_m", m["wave_mean_m"]),
                         ("wave_period_s", m["wave_period_s"]), ("precip_mm", m["precip_mm"]),
                         ("cloud_pct", m["cloud_mean_pct"]), ("moonlight", m["moonlight"]),
                         ("temp_c", rnd(s_wx.get("temperature_2m", t))), ("pressure_hpa", m["pressure_hpa"])):
                cols[k].append(v)
            t += timedelta(hours=1)
        timeline[p["id"]] = cols

    # ------------------------------------------------ точки и SST
    names = {p["id"]: p["name"] for p in cfg["coastal"] + cfg["marine"]}
    for r in cfg["rivers"]:
        for c in r["catchment"]:
            names[c["id"]] = f'{r["name"]} ({c["zone"]})'

    def point_info(kind, pid, lat, lon, resp):
        return {"id": pid, "name": names.get(pid), "kind": kind, "lat": lat, "lon": lon,
                "model_grid": {"lat": rnd(resp.get("latitude"), 4), "lon": rnd(resp.get("longitude"), 4)},
                "elevation_m": resp.get("elevation")}
    points_out = [point_info(*p, x) for p, x in zip(fpoints, fc)] + \
                 [point_info(*p, x) for p, x in zip(mpoints, mc)]

    sst = {}
    for pid, s in sea.items():
        v = s.get("sea_surface_temperature", now_hour)
        if v is None:
            warn(f"{pid}: нет sea_surface_temperature на текущий час")
        sst[pid] = v

    latest = {
        "generated_at": now.isoformat(timespec="minutes"),
        "timezone": "Europe/Istanbul",
        "units": {"wind": "m/s", "wave": "m", "precip": "mm", "pressure": "hPa", "sst": "°C"},
        "source": "Open-Meteo.com (CC BY 4.0), best_match",
        "source_model_runs": runs,
        "points": points_out,
        "moon": moon_phase(now),
        "rivers": rivers_out,
        "windows": windows,
        "timeline": timeline,
        "sea_surface_temperature_now": sst,
        "warnings": WARNINGS,
    }

    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "history"), exist_ok=True)
    with open(os.path.join(DATA, "raw", "latest_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"fetched_at": latest["generated_at"], "forecast": fc, "marine": mc}, f, ensure_ascii=False)
    # Компактно, без отступов: с почасовой лентой на 4 точки файл с отступами
    # раздувается втрое, а снимок в history пишется каждый день.
    body = json.dumps(latest, ensure_ascii=False, separators=(",", ":"))
    for path in (os.path.join(DATA, "latest.json"),
                 os.path.join(DATA, "history", now.strftime("%Y-%m-%d") + ".json")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
    with open(os.path.join(DATA, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_summary(latest, now))
    print(f"OK: {len(windows)} окон, {len(rivers_out)} устьев, {len(WARNINGS)} предупреждений")
    return 0


# ---------------------------------------------------------------- summary.md

WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def short(w):
    d = datetime.fromisoformat(w["date"])
    if w["kind"] == "ночь":
        # Ночь привязана к дате вечера; для человека понятнее «ночь на <следующий день>».
        n = d + timedelta(days=1)
        return f"ночь на {WD[n.weekday()]} {n.day:02d}.{n.month:02d} {w['start'][11:16]}–{w['end'][11:16]}"
    return f"{WD[d.weekday()]} {d.day:02d}.{d.month:02d} {w['kind']} {w['start'][11:16]}–{w['end'][11:16]}"


def yn(b):
    return "да" if b else "нет"


def render_summary(L, now):
    out = [f"# Леврек: сводка на {now.strftime('%d.%m.%Y %H:%M')} (Стамбул)", ""]
    m = L["moon"]
    names = {p["id"]: p["name"] for p in L["points"]}
    sst = ", ".join(f"{names.get(k, k).lower()} {v} °C" for k, v in L["sea_surface_temperature_now"].items() if v is not None)
    out += [f"Луна: {m['phase']}, {m['illumination_pct']}% (полнолуние {m['next_full_moon']}, новолуние {m['next_new_moon']}). "
            f"Вода: {sst}.", ""]

    out += ["## Топ-3 окна на 7 дней", ""]
    cand = [w for w in L["windows"] if not w["no_go"]]
    top = sorted(cand, key=lambda w: (-w["score"], w["start"]))[:3]
    if not top:
        out.append("Подходящих окон нет.")
    for i, w in enumerate(top, 1):
        extra = " ⚠ опасно на камнях" if w["dangerous_on_rocks"] else ""
        out.append(f"{i}. **{w['score']}/5 — {w['point_name']}, {short(w)}**{extra}. "
                   f"Ветер {w['wind_dir']} {w['wind_mean_ms']} м/с (порывы {w['gust_max_ms']}), "
                   f"волна {w['wave_mean_m']} м / {w['wave_period_s']} с. "
                   f"Почему: {'; '.join(w['reasons'][1:]) or 'ничего особенного'}.")
    out.append("")

    out += ["## Окна на ближайшие 3 дня", "",
            "| Окно | Место | Оценка | Ветер, м/с | Порыв | Нагон | Волна ср/макс, м | Период, с | Осадки, мм | Гроза | Давл. 6ч, гПа |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    limit = (now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M")
    for w in L["windows"]:
        if w["start"] > limit:
            continue
        sc = f"{w['score']}" + (" no_go" if w["no_go"] else "") + (" ⚠камни" if w["dangerous_on_rocks"] else "")
        tr = w["pressure_trend_6h_hpa"]
        out.append(f"| {short(w)} | {w['point_name']} | {sc} | {w['wind_dir']} {w['wind_mean_ms']} | {w['gust_max_ms']} | "
                   f"{yn(w['onshore'])} | {w['wave_mean_m']}/{w['wave_max_m']} | {w['wave_period_s']} | {w['precip_mm']} | "
                   f"{yn(w['thunder'])} | {'' if tr is None else f'{tr:+.1f}'} |")
    out.append("")

    out += ["## Устья", "",
            "| Водоток | Состояние | 24 ч, мм | 48 ч, мм | Прогноз 24/48 ч, мм | Пик за 72 ч | С пика, ч |",
            "|---|---|---|---|---|---|---|"]
    for r in L["rivers"]:
        peak = "—" if not r["peak_72h_time"] else f"{r['peak_72h_time'][5:].replace('T', ' ')} ({r['peak_72h_mm_per_h']} мм/ч)"
        out.append(f"| {r['name']} | {r['mouth_state_ru']} | {r['past_24h_mm']} | {r['past_48h_mm']} | "
                   f"{r['next_24h_mm']}/{r['next_48h_mm']} | {peak} | {'—' if r['hours_since_peak'] is None else int(r['hours_since_peak'])} |")
    for r in L["rivers"]:
        if r.get("regulated"):
            z = r.get("zones", {})
            b, a = z.get("below_dam", {}), z.get("above_dam", {})
            out.append("")
            out.append(f"{r['name']}: сток частично регулируется плотиной Dim Barajı. "
                       f"Ниже плотины за 48 ч {b.get('past_48h_mm')} мм, выше — {a.get('past_48h_mm')} мм.")
    if L["warnings"]:
        out += ["", "## Предупреждения", ""] + [f"- {w}" for w in L["warnings"]]
    out += ["", "Данные: [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
