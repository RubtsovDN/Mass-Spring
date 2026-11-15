"""ЛР 3. Quanser 2-DOF: загрузка, линейная механика, SMC, графы, RealNVP.

Запуск: import lab3_quanser_core as core. Все расчёты здесь, ноутбук — визуализация.
Данные никогда не подменяются искусственными, если API Harvard Dataverse недоступен.
"""
from __future__ import annotations

import re
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal.windows import hann
from scipy.stats import ks_2samp

DOI = "doi:10.7910/DVN/XGTWSJ"
API = "https://dataverse.harvard.edu/api"
# Номинальные значения из статьи 10.1155/2014/529293, таблица 1.
MASSES = (2.45, 1.0)  # kg: body, wheel
NOMINAL = np.array([900.0, 2500.0, 7.5, 5.0], dtype=float)  # k_s,k_t,c_s,c_t
PARAMS = ("k_s", "k_t", "c_s", "c_t")
# Индикатор включения физического канала связи в структурный граф.
ALL_MASKS = np.array([[bool((i >> j) & 1) for j in range(4)] for i in range(16)], dtype=bool)


@dataclass
class Experiment:
    t: np.ndarray
    x_s: np.ndarray
    x_u: np.ndarray
    road: np.ndarray
    profile: str
    info: str


@dataclass
class Spectrum:
    omega: np.ndarray
    H: np.ndarray  # shape (n_freq,2); body and tire transfer functions
    weight: np.ndarray  # excitation-aware spectral weight


# Версия 1.0 (Salah, 2025), метаданные Harvard Dataverse.
# Прямые ID позволяют не обращаться к /api/datasets/:persistentId/,
# который может отвечать 403 на некоторых сетях.
FILE_IDS = {
    "Body Displacement Data.xlsx": 12133334,
    "Tire Position Data.xlsx": 12133331,
    "Vertical Displacement Error.xlsx": 12133329,
}
FILE_MD5 = {
    "Body Displacement Data.xlsx": "4f5e684f8d128b09a0cb631f011375f7",
    "Tire Position Data.xlsx": "4005e00bc9094988bdae73240b6ad9a7",
    "Vertical Displacement Error.xlsx": "e5f13bc25a55562b44dbf47d12664ae5",
}
DATASET_PAGE = "https://doi.org/10.7910/DVN/XGTWSJ"


def _validate_quanser_file(path: str | Path, name: str) -> None:
    """Убедиться, что это именно исходный XLSX Harvard, а не страница ошибки/тест."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 10_000:
        raise ValueError(f"{path} отсутствует или подозрительно мал")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name}: скачан не XLSX (возможно, HTML-страница 403)")
    expected = FILE_MD5.get(name)
    if expected:
        digest = hashlib.md5()  # Сверка опубликованной контрольной суммы, не криптография.
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected:
            raise ValueError(
                f"{path.name}: MD5 не совпал с оригинальным файлом Harvard Dataverse. "
                "Скачайте исходный XLSX без изменений, не пересохраняйте в Excel."
            )


def fetch_quanser(destination: str | Path, names: tuple[str, ...] = (
    "Body Displacement Data.xlsx", "Tire Position Data.xlsx", "Vertical Displacement Error.xlsx"
)) -> dict[str, Path]:
    """Загрузка опубликованных файлов Quanser или использование локального кеша.

    В отличие от старой версии НЕ обращается к API метаданных, который может
    отвечать HTTP 403: известные публичные file ID подставляются напрямую.
    Если прямые HTTP-запросы также запрещены, пользователь скачивает XLSX
    через браузер в папку data/quanser; при повторном запуске сети не требуется.
    Защита: проверяются ZIP-формат и опубликованные MD5-суммы настоящих XLSX.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    need_download = []
    for name in names:
        if name not in FILE_IDS:
            raise KeyError(f"Неизвестный файл {name!r}; допустимы {list(FILE_IDS)}")
        path = dest / name
        if path.exists():
            _validate_quanser_file(path, name)
            files[name] = path
        else:
            need_download.append(name)
    if not need_download:
        return files

    errors = []
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; ResearchNotebook/1.0)",
            "Accept": "application/octet-stream,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
            "Referer": "https://dataverse.harvard.edu/",
        })
        for name in need_download:
            path = dest / name
            url = f"{API}/access/datafile/{FILE_IDS[name]}"
            temp = dest / (name + ".part")
            try:
                with session.get(url, stream=True, timeout=(15, 80),
                                 allow_redirects=True) as response:
                    response.raise_for_status()
                    with temp.open("wb") as out:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out.write(chunk)
                _validate_quanser_file(temp, name)
                temp.replace(path)
                files[name] = path
            except (requests.RequestException, OSError, ValueError) as exc:
                temp.unlink(missing_ok=True)
                errors.append(f"{name}: {exc}")
                # Если сайт отклоняет запросы, другие файлы могут быть доступны
                # в локальном кеше/через браузер, поэтому перечисляем все ошибки.

    if errors:
        link_lines = "\n".join(
            f"  {name}: {API}/access/datafile/{FILE_IDS[name]}"
            for name in need_download if name not in files
        )
        raise RuntimeError(
            "Автоматическая загрузка публичных XLSX Quanser не удалась. "
            "Harvard Dataverse может отвечать 403 на программные запросы.\n"
            "Надёжный способ: откройте набор в БРАУЗЕРЕ, скачайте оригинальные файлы "
            "и положите их в папку рядом с ноутбуком:\n"
            f"  {dest.resolve()}\n"
            "Имена должны быть точными, содержимое Excel НЕ пересохраняйте. "
            "После этого снова запустите ячейку: будут прочитаны локальные XLSX.\n"
            f"Страница набора: {DATASET_PAGE}\n"
            "Прямые ссылки на отсутствующие файлы:\n" + link_lines +
            "\nОшибки автоматической загрузки:\n" + "\n".join(errors)
        )
    return files


def _normal(s) -> str:
    return re.sub(r"[^a-z0-9а-я]+", " ", str(s).strip().lower()).strip()


def _profile_found(s: str, profile: str) -> bool:
    s, p = _normal(s), _normal(profile)
    if "variable" in p:
        return ("variab" in s or "перемен" in s) and ("freq" in s or "частот" in s)
    if "sine" in p:
        return any(x in s for x in ("sine", "sinus", "синус"))
    if p == "step":
        return s == "step" or s == "step input"
    return p in s


def _numeric_column(raw: pd.DataFrame, col: int, start_row: int) -> np.ndarray:
    return pd.to_numeric(raw.iloc[start_row:, col], errors="coerce").to_numpy(dtype=float)


def _monotone_time(t: np.ndarray) -> bool:
    t = t[np.isfinite(t)]
    return len(t) > 30 and np.all(np.diff(t) > 0) and np.median(np.diff(t)) > 0 and t[-1]-t[0] > 0.1


def select_excel_series(path: str | Path, profile: str = "Variable frequency", mode: str = "Passive",
                        override: dict | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """По двухуровневым заголовкам определяет колонку profile/mode и колонку времени.

    override={'sheet': 0, 'time_col': 0, 'value_col': 1, 'start_row': 2}
    предусмотрен для другого размещения заголовков; никакой выбор по номеру
    колонки без проверки заголовков/монотонности времени не выполняется.
    """
    path = Path(path)
    xls = pd.ExcelFile(path)
    preview = []
    for sheet in ([override.get("sheet", 0)] if override else xls.sheet_names):
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        if override:
            tc, vc, sr = (int(override[k]) for k in ("time_col", "value_col", "start_row"))
            t, y = _numeric_column(raw, tc, sr), _numeric_column(raw, vc, sr)
            use = np.isfinite(t) & np.isfinite(y)
            if not _monotone_time(t[use]):
                raise ValueError(f"Указанный вручную столбец времени {tc} не монотонный: {path.name}")
            return t[use], y[use], f"{path.name}: sheet={sheet}, time_col={tc}, value_col={vc} [override]"
        scan = raw.iloc[:min(18, len(raw)), :].fillna("")
        preview.append((str(sheet), scan.iloc[:6, :min(18, len(scan.columns))].to_string(index=False, header=False)))
        header = [[_normal(scan.iat[r, c]) for c in range(scan.shape[1])] for r in range(scan.shape[0])]
        profile_marks = [(r, c) for r in range(len(header)) for c, s in enumerate(header[r])
                         if _profile_found(s, profile)]
        # Профиль может быть задан названием листа, а не верхней строкой.
        if not profile_marks and _profile_found(str(sheet), profile):
            profile_marks = [(0, 0)]
        if not profile_marks:
            continue
        for pr, pc in profile_marks:
            # Если 5 профилей представлены горизонтальными блоками, найдём правую границу.
            other = [c for r, c in [(rr, cc) for rr in range(len(header)) for cc, s in enumerate(header[rr])
                                 if rr <= pr+2 and any(v in s for v in ("step", "stairs", "sine", "variab", "bump"))]
                     if c > pc]
            right = min(other, default=scan.shape[1])
            candidates = [(r, c) for r in range(len(header)) for c in range(pc, right)
                          if _normal(mode) in header[r][c] and header[r][c] != ""]
            if not candidates:
                # Крайний случай: профиль — название листа, а режим — в первой строке.
                candidates = [(r, c) for r in range(len(header)) for c in range(scan.shape[1])
                              if _normal(mode) in header[r][c]]
            for hr, vc in candidates:
                sr = max(pr, hr) + 1
                possible_time_cols = [(r, c) for r in range(min(sr, len(header))) for c in range(scan.shape[1])
                                      if (header[r][c] in ("time", "t", "time s", "time sec", "time second", "время"))]
                # Перед PASSIVE обычно расположен Time, иногда единственный общий Time в начале листа.
                possible_time_cols = sorted(possible_time_cols, key=lambda q: (
                    0 if pc <= q[1] < right else 1,
                    abs(vc-q[1]) + (0 if q[1] <= vc else 100), abs(q[0]-hr)
                ))
                if not possible_time_cols:
                    # В таблицах без текстового "Time" ищем только строго монотонную числовую колонку слева.
                    possible_time_cols = [(sr-1, c) for c in range(max(0, vc-5), vc)]
                for _, tc in possible_time_cols:
                    if tc == vc:
                        continue
                    t, y = _numeric_column(raw, tc, sr), _numeric_column(raw, vc, sr)
                    use = np.isfinite(t) & np.isfinite(y)
                    if len(t[use]) >= 40 and _monotone_time(t[use]):
                        return t[use], y[use], f"{path.name}: sheet={sheet}, time_col={tc}, value_col={vc}"
    detail = "\n".join(f"Лист {sheet}:\n{view}" for sheet, view in preview[:2])
    raise ValueError(f"Не удалось определить {profile}/{mode} в {path.name}. "
                     "Ни одна пара (время, значение) не прошла проверку.\n"
                     "Проверьте заголовки ниже и задайте COLUMN_OVERRIDES в ноутбуке.\n" + detail)


def make_experiment(files: dict[str, Path], profile="Variable frequency", mode="Passive",
                    road_sign=+1, overrides=None, n_points=2400) -> Experiment:
    """Формирует синхронные наблюдения; r = x_s + sign * (vertical displacement error).

    sign=+1: e=r-x_s, sign=-1: e=x_s-r. Знак необходимо сверить с метаданными
    первичного эксперимента: определение `difference` не фиксирует ориентацию.
    """
    keys = ("Body Displacement Data.xlsx", "Tire Position Data.xlsx", "Vertical Displacement Error.xlsx")
    series = []
    notes = []
    for name in keys:
        t, y, note = select_excel_series(files[name], profile=profile, mode=mode,
                                          override=(overrides or {}).get(name))
        series.append((t, y))
        notes.append(note)
    lo = max(a[0][0] for a in series)
    hi = min(a[0][-1] for a in series)
    if hi <= lo or hi-lo < 0.2:
        raise ValueError("Временные интервалы трёх файлов не пересекаются")
    # Честная интерполяция: сообщаем интервал и число отсчётов; не соединяем разные испытания.
    n = min(n_points, min(len(t) for t, _ in series))
    if n < 50:
        raise ValueError("Слишком мало общих отсчётов")
    t_common = np.linspace(lo, hi, n)
    body, wheel, err = [np.interp(t_common, t, y) for t, y in series]
    road = body + int(road_sign) * err
    if not np.isfinite([body, wheel, road]).all():
        raise ValueError("В данных есть NaN/inf после синхронизации")
    if np.std(road) < 1e-8:
        raise ValueError("Восстановленное дорожное воздействие почти постоянно; проверьте road_sign")
    return Experiment(t_common, body, wheel, road, profile,
                      "\n".join(notes) + f"; road = body + ({road_sign})*error")


def make_spectrum(exp: Experiment, n_freq=24, low_hz=0.35, high_hz=12.0) -> Spectrum:
    """Комплексные частотные отклики из реальной записи; оконное FFT, без синтетики."""
    n = len(exp.t)
    dt = np.median(np.diff(exp.t))
    if not np.allclose(np.diff(exp.t), dt, atol=max(1e-10, dt*0.005)):
        raise ValueError("Для FFT требуется равномерная временная сетка")
    win = hann(n, sym=False)
    signals = [exp.road, exp.x_s, exp.x_u]
    r, xs, xu = [np.fft.rfft((s-np.mean(s))*win) for s in signals]
    hz = np.fft.rfftfreq(n, dt)
    valid = np.where((hz >= low_hz) & (hz <= high_hz) &
                     (np.abs(r) > max(1e-12, 0.015*np.max(np.abs(r)))))[0]
    if len(valid) < 4:
        raise ValueError(f"Недостаточно информативных частот (найдено {len(valid)}): "
                         "проверьте профиль, знак дорожного сигнала и диапазон частот")
    chosen = valid[np.argsort(np.abs(r[valid]))[-min(n_freq, len(valid)):]]
    chosen.sort()
    H = np.column_stack((xs[chosen]/r[chosen], xu[chosen]/r[chosen]))
    weight = np.abs(r[chosen])**2
    weight = np.maximum(0.02, weight/np.max(weight))
    return Spectrum(2*np.pi*hz[chosen], H, weight)


def split_spectrum(s: Spectrum) -> tuple[Spectrum, Spectrum]:
    idx = np.arange(len(s.omega))
    train = idx[idx % 3 != 0]
    valid = idx[idx % 3 == 0]
    if len(valid) < 1 or len(train) < 3:
        raise ValueError("Недостаточно информативных частот для разбиения")
    return Spectrum(s.omega[train], s.H[train], s.weight[train]), Spectrum(s.omega[valid], s.H[valid], s.weight[valid])


def response(omega: np.ndarray, params: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """H=(X_s/R, X_u/R) для 2-DOF. Batch: params [N,4] -> [N,F,2]."""
    par = np.asarray(params, float)
    one = par.ndim == 1
    par = np.atleast_2d(par)
    if par.shape[1] != 4:
        raise ValueError("Ожидаются четыре параметра k_s,k_t,c_s,c_t")
    if mask is not None:
        par = par * np.asarray(mask, float)[None, :]
    ks, kt, cs, ct = (par[:, i, None] for i in range(4))
    w = np.asarray(omega, float)[None, :]
    ms, mu = MASSES
    q = ks + 1j*w*cs
    p = kt + 1j*w*ct
    a = q - ms*w*w
    d = q + p - mu*w*w
    det = a*d - q*q
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xs = q*p/det
        xu = a*p/det
    out = np.stack((xs, xu), axis=-1)
    # Часть графов физически вырожденная; вместо NaN отдаём нулевой отклик.
    out = np.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
    return out[0] if one else out


def relative_error(spectrum: Spectrum, params: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = response(spectrum.omega, params, mask)
    num = np.sum(spectrum.weight[:, None] * np.abs(pred-spectrum.H)**2)
    den = np.sum(spectrum.weight[:, None] * np.abs(spectrum.H)**2) + 1e-10
    return float(np.sqrt(num / den))


def _complex_residual(spectrum: Spectrum, params: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = (response(spectrum.omega, params, mask)-spectrum.H) * np.sqrt(spectrum.weight[:, None])
    scale = max(0.1, np.sqrt(np.mean(np.abs(spectrum.H)**2)))
    return np.r_[delta.real.ravel(), delta.imag.ravel()] / scale


def fit_graph(spectrum: Spectrum, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    params = np.zeros(4)
    if not mask.any():
        return params
    idx = np.flatnonzero(mask)
    def loss(logf):
        trial = np.zeros(4)
        trial[idx] = NOMINAL[idx] * np.exp(logf)
        return _complex_residual(spectrum, trial, mask)
    result = least_squares(loss, np.zeros(len(idx)), bounds=(-np.log(5)*np.ones(len(idx)),
                         np.log(5)*np.ones(len(idx))), max_nfev=65)
    params[idx] = NOMINAL[idx]*np.exp(result.x)
    return params


def _logprior(logp: np.ndarray) -> np.ndarray:
    diff = (logp - np.log(NOMINAL)[None, :]) / 0.50
    ok = np.all(np.abs(logp-np.log(NOMINAL)[None, :]) <= 1.6, axis=1)
    return np.where(ok, -0.5*np.sum(diff*diff, axis=1), -1e20)


def _loglike(logp: np.ndarray, s: Spectrum) -> np.ndarray:
    h = response(s.omega, np.exp(logp))
    d = h - s.H[None, :, :]
    scale = np.maximum(0.08, 0.18*np.sqrt(np.mean(np.abs(s.H)**2, axis=0)))
    return -0.5 * np.sum(s.weight[None,:,None] * (np.abs(d)/scale[None,None,:])**2, axis=(1,2))


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(w)
    u = (rng.random() + np.arange(n))/n
    return np.searchsorted(np.cumsum(w), u, side="right").clip(max=n-1)


def smc(spectrum: Spectrum, n_particles=240, seed=42, n_temper=14, mh_steps=2) -> np.ndarray:
    """Tempered Sequential Monte Carlo по 4 параметрам, MH-rejuvenation в log-space."""
    rng = np.random.default_rng(seed)
    lp = np.log(NOMINAL)[None, :] + rng.normal(0, .45, (n_particles, 4))
    ll = _loglike(lp, spectrum)
    weights = np.ones(n_particles)/n_particles
    beta_prev = 0.0
    for beta in np.linspace(0, 1, n_temper+1)[1:]:
        lw = np.log(weights+1e-300) + (beta-beta_prev)*ll
        lw -= np.max(lw)
        weights = np.exp(lw)
        weights /= weights.sum()
        if 1 / np.sum(weights**2) < 0.75*n_particles:
            idx = systematic_resample(weights, rng)
            lp, ll = lp[idx], ll[idx]
            weights[:] = 1/n_particles
        for _ in range(mh_steps):
            proposed = lp + rng.normal(0, .09, lp.shape)
            new_ll = _loglike(proposed, spectrum)
            loga = _logprior(proposed) - _logprior(lp) + beta*(new_ll - ll)
            accept = np.log(rng.random(n_particles)) < loga
            lp[accept], ll[accept] = proposed[accept], new_ll[accept]
        beta_prev = beta
    idx = systematic_resample(weights, rng)
    return np.exp(lp[idx])


def compare_posteriors(p0: np.ndarray, p1: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(PARAMS):
        test = ks_2samp(p0[:, j], p1[:, j], method="auto")
        rows.append({"parameter": name, "median_original": np.median(p0[:, j]),
                     "median_zeroed": np.median(p1[:, j]), "KS_D": test.statistic,
                     "KS_pvalue": test.pvalue})
    return pd.DataFrame(rows)


def score_candidates(train: Spectrum, holdout: Spectrum, ks_df: pd.DataFrame,
                     complexity_weight=.025, intervention_weight=.07) -> pd.DataFrame:
    """Точность на невиденных частотах + число каналов + штраф за удаление KS-значимых каналов."""
    ks_vals = ks_df.set_index("parameter").loc[list(PARAMS), "KS_D"].to_numpy()
    # KS — не причинный эффект; это эвристический вес сохранения физической связи.
    importance = ks_vals / (ks_vals.sum() + 1e-12)
    results = []
    for mask in ALL_MASKS:
        fitted = fit_graph(train, mask)
        train_err = relative_error(train, fitted)
        valid_err = relative_error(holdout, fitted)
        penalty = float(np.dot(importance, 1-mask.astype(float)))
        score = valid_err + complexity_weight*mask.sum() + intervention_weight*penalty
        results.append({"mask": "".join(str(int(x)) for x in mask),
                        "k_s": fitted[0], "k_t": fitted[1], "c_s": fitted[2], "c_t": fitted[3],
                        "train_nrmse": train_err, "test_nrmse": valid_err,
                        "edges": int(mask.sum()), "intervention_penalty": penalty, "J": score})
    return pd.DataFrame(results).sort_values("J").reset_index(drop=True)


def mask_to_array(code: str) -> np.ndarray:
    if len(code) != 4 or set(code)-set("01"):
        raise ValueError("Маска из четырёх двоичных символов")
    return np.array([x == "1" for x in code], bool)


def dynamics_graph(mask: np.ndarray) -> nx.DiGraph:
    """DAG одного перехода t->t+dt (узлы ускорений — следующий слой времени)."""
    b = np.asarray(mask, bool)
    g = nx.DiGraph()
    g.add_nodes_from(["x_s(t)", "v_s(t)", "x_u(t)", "v_u(t)", "r(t)", "r'(t)",
                      "a_s(t+dt)", "a_u(t+dt)"])
    pairs = [
        (("x_s(t)","a_s(t+dt)"),("x_u(t)","a_s(t+dt)"),
         ("x_s(t)","a_u(t+dt)"),("x_u(t)","a_u(t+dt)")),
        (("x_u(t)","a_u(t+dt)"),("r(t)","a_u(t+dt)")),
        (("v_s(t)","a_s(t+dt)"),("v_u(t)","a_s(t+dt)"),
         ("v_s(t)","a_u(t+dt)"),("v_u(t)","a_u(t+dt)")),
        (("v_u(t)","a_u(t+dt)"),("r'(t)","a_u(t+dt)")),
    ]
    for k, enabled in enumerate(b):
        if enabled:
            for source, target in pairs[k]:
                g.add_edge(source, target, label=PARAMS[k])
    return g


def simulate(exp: Experiment, theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Time-domain ODE validation; starts from observed initial position and estimated velocity."""
    ks, kt, cs, ct = np.asarray(theta)*np.asarray(mask, float)
    ms, mu = MASSES
    t, road = exp.t, exp.road
    dr = np.gradient(road, t)
    def deriv(t0, y):
        xs, xu, vs, vu = y
        r = float(np.interp(t0, t, road))
        vr = float(np.interp(t0, t, dr))
        f = ks*(xs-xu) + cs*(vs-vu)
        return [vs, vu, -f/ms, (f-kt*(xu-r)-ct*(vu-vr))/mu]
    vs0 = np.gradient(exp.x_s, t)[0]
    vu0 = np.gradient(exp.x_u, t)[0]
    result = solve_ivp(deriv, (t[0], t[-1]), [exp.x_s[0], exp.x_u[0], vs0, vu0],
                       t_eval=t, rtol=2e-5, atol=2e-7)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


# ---------- Нормализующий поток для бинарных шаблонов графов ----------
# Дискретные графы кодируются как 4-битовые маски; поток учится на де-квантизованных
# 4D векторах, после генерации применяется внешний порог (необратимая операция).

def learn_graph_flow(score_table: pd.DataFrame, epochs=350, n_samples=200, seed=42):
    """Обучение маленького RealNVP на лучших графах; возвращает поток и прогресс семплирования."""
    import torch
    from torch import nn
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class Coupling(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
            self.net = nn.Sequential(nn.Linear(4, 24), nn.Tanh(), nn.Linear(24, 24),
                                     nn.Tanh(), nn.Linear(24, 8))
        def forward(self, z):
            masked = z*self.mask
            s, t = self.net(masked).chunk(2, dim=-1)
            s = 0.65*torch.tanh(s)
            x = masked + (1-self.mask)*(z*torch.exp(s)+t)
            return x, (s*(1-self.mask)).sum(-1)
        def inverse(self, x):
            masked = x*self.mask
            s, t = self.net(masked).chunk(2, dim=-1)
            s = 0.65*torch.tanh(s)
            z = masked + (1-self.mask)*(x-t)*torch.exp(-s)
            return z, -(s*(1-self.mask)).sum(-1)

    class RealNVP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Coupling(m) for m in
                     ([1,0,1,0], [0,1,0,1], [1,1,0,0], [0,0,1,1])])
        def forward(self, z):
            ld = z.new_zeros(z.shape[0])
            for layer in self.layers:
                z, change = layer(z)
                ld = ld + change
            return z, ld
        def log_prob(self, x):
            logdet = x.new_zeros(x.shape[0])
            for layer in reversed(self.layers):
                x, change = layer.inverse(x)
                logdet = logdet + change
            return -0.5*(x*x).sum(-1) - 0.5*4*np.log(2*np.pi) + logdet
        @torch.no_grad()
        def sample(self, n):
            x, _ = self.forward(torch.randn(n, 4))
            return (x.numpy() > 0).astype(int)

    df = score_table.sort_values("J").copy()
    best_codes = df["mask"].head(min(5, len(df))).to_list()
    weights = np.exp(-8*(df["J"].head(len(best_codes)).to_numpy() - df["J"].iloc[0]))
    weights /= weights.sum()
    best_bits = np.array([[int(c) for c in code] for code in best_codes])
    flow = RealNVP()
    optim = torch.optim.Adam(flow.parameters(), lr=.002)
    loss_history = []
    sample_history = []
    # Отобразим поколение до обучения, в середине и после обучения.
    for epoch in range(epochs+1):
        if epoch in (0, epochs//2, epochs):
            sample_history.append((epoch, flow.sample(n_samples)))
        if epoch == epochs:
            break
        ids = rng.choice(len(best_bits), 128, p=weights)
        noise = rng.normal(0, .3, (128,4))
        x = torch.tensor((2*best_bits[ids]-1)*1.25 + noise, dtype=torch.float32)
        loss = -flow.log_prob(x).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_history.append(float(loss.detach()))
    return flow, np.asarray(loss_history), sample_history


def graph_sample_summary(history: list[tuple[int, np.ndarray]], table: pd.DataFrame) -> pd.DataFrame:
    score_map = dict(zip(table["mask"], table["J"]))
    records = []
    for epoch, bits in history:
        scores = [score_map["".join(map(str, row))] for row in bits]
        records.append({"epoch": epoch, "mean_J": float(np.mean(scores)),
                        "median_J": float(np.median(scores)),
                        "best_J": float(np.min(scores)),
                        "unique_graphs": len(set("".join(map(str, row)) for row in bits))})
    return pd.DataFrame(records)


# ---------- Дополнения: анализ регуляризации и вычислительный граф ----------

def rescore_graphs(score_table: pd.DataFrame,
                   complexity_weight: float,
                   intervention_weight: float) -> pd.DataFrame:
    """Быстро переоценивает графы при новых весах, не подбирая параметры повторно.

    test_nrmse, edges и intervention_penalty уже вычислены на одних и тех же
    обучающих/проверочных данных; менять их при изменении весов не требуется.
    """
    if complexity_weight < 0 or intervention_weight < 0:
        raise ValueError("Веса регуляризации должны быть неотрицательными")
    required = {"mask", "test_nrmse", "edges", "intervention_penalty"}
    missing = required.difference(score_table.columns)
    if missing:
        raise KeyError(f"Нет столбцов для пересчёта: {sorted(missing)}")
    df = score_table.copy()
    df["J"] = (df["test_nrmse"]
               + complexity_weight * df["edges"]
               + intervention_weight * df["intervention_penalty"])
    return df.sort_values(["J", "test_nrmse", "edges", "mask"],
                          ascending=[True, True, True, True],
                          kind="mergesort").reset_index(drop=True)


def regularization_sweep(score_table: pd.DataFrame,
                         complexity_weights,
                         intervention_weights) -> pd.DataFrame:
    """Оптимальная маска для каждой пары весов (полный перебор 16 структур)."""
    rows = []
    for w_inter in intervention_weights:
        for w_comp in complexity_weights:
            ranked = rescore_graphs(score_table, float(w_comp), float(w_inter))
            best = ranked.iloc[0]
            rows.append({"lambda_complexity": float(w_comp),
                         "mu_intervention": float(w_inter),
                         "best_mask": str(best["mask"]),
                         "test_nrmse": float(best["test_nrmse"]),
                         "edges": int(best["edges"]),
                         "J": float(best["J"])})
    return pd.DataFrame(rows)


def computation_graph(mask: np.ndarray) -> nx.DiGraph:
    """Вычислительный DAG правой части ODE: входы -> разности -> силы -> производные.

    В отличие от dynamics_graph, это граф **арифметических операций**,
    а не упрощённых зависимостей входных и выходных координат.
    """
    b = np.asarray(mask, dtype=bool)
    if b.shape != (4,):
        raise ValueError("Маска должна иметь длину 4: k_s,k_t,c_s,c_t")
    g = nx.DiGraph()
    inputs = {"x_s": r"$x_s$", "x_u": r"$x_u$", "v_s": r"$v_s$",
              "v_u": r"$v_u$", "r": r"$r$", "v_r": r"$\dot r$"}
    outputs = {"dx_s": r"$\dot x_s$", "dx_u": r"$\dot x_u$",
               "a_s": r"$a_s$", "a_u": r"$a_u$"}
    for node, label in inputs.items():
        g.add_node(node, kind="input", label=label)
    for node, label in outputs.items():
        g.add_node(node, kind="output", label=label)
    g.add_edge("v_s", "dx_s")
    g.add_edge("v_u", "dx_u")
    terms = [
        # coefficient, difference, inputs, force label, accelerations
        ("k_s", "d_xs", ("x_s", "x_u"), "force_ks", ("a_s", "a_u"),
         r"$x_s-x_u$", r"$k_s\Delta x_s$"),
        ("k_t", "d_xt", ("x_u", "r"), "force_kt", ("a_u",),
         r"$x_u-r$", r"$k_t\Delta x_t$"),
        ("c_s", "d_vs", ("v_s", "v_u"), "force_cs", ("a_s", "a_u"),
         r"$v_s-v_u$", r"$c_s\Delta v_s$"),
        ("c_t", "d_vt", ("v_u", "v_r"), "force_ct", ("a_u",),
         r"$v_u-\dot r$", r"$c_t\Delta v_t$")
    ]
    for active, entry in zip(b, terms):
        if not active:
            continue
        coef, diff, inputs2, force, accs, d_label, f_label = entry
        g.add_node(coef, kind="parameter", label=f"${coef}$")
        g.add_node(diff, kind="difference", label=d_label)
        g.add_node(force, kind="force", label=f_label)
        for source in inputs2:
            g.add_edge(source, diff)
        g.add_edge(diff, force)
        g.add_edge(coef, force)
        for a in accs:
            if a == "a_s":
                sign = "-"
                mass = "m_s"
            else:
                sign = "+" if coef in ("k_s", "c_s") else "-"
                mass = "m_u"
            g.add_edge(force, a, label=f"{sign} / {mass}")
    assert nx.is_directed_acyclic_graph(g)
    return g


def draw_computation_graph(mask: np.ndarray, ax=None, title=None):
    """Рисует вычислительный граф расчёта производных 2-DOF Mass–Spring–Damper.

    Возвращает (fig, ax); поддерживает отдельную панель в subplot.
    """
    import matplotlib.pyplot as plt
    g = computation_graph(mask)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12.0, 5.5))
    else:
        fig = ax.figure
    positions = {
        "x_s": (0.0, 7.0), "x_u": (0.0, 5.8),
        "v_s": (0.0, 4.6), "v_u": (0.0, 3.4),
        "r": (0.0, 2.2), "v_r": (0.0, 1.0),
        "d_xs": (2.1, 7.0), "d_vs": (2.1, 5.2),
        "d_xt": (2.1, 3.4), "d_vt": (2.1, 1.6),
        "k_s": (3.3, 7.85), "c_s": (3.3, 6.0),
        "k_t": (3.3, 4.15), "c_t": (3.3, 2.30),
        "force_ks": (4.7, 7.0), "force_cs": (4.7, 5.2),
        "force_kt": (4.7, 3.4), "force_ct": (4.7, 1.6),
        "dx_s": (8.0, 7.5), "dx_u": (8.0, 6.3),
        "a_s": (8.0, 4.7), "a_u": (8.0, 2.5),
    }
    palette = {"input": "#dbeafe", "parameter": "#fef3c7",
               "difference": "#ecfccb", "force": "#fce7f3", "output": "#dcfce7"}
    grouped = {}
    for node, d in g.nodes(data=True):
        grouped.setdefault(d["kind"], []).append(node)
    for kind, nodes in grouped.items():
        nx.draw_networkx_nodes(
            g, positions, nodelist=nodes, ax=ax, node_color=palette[kind],
            node_size=(1250 if kind in ("force", "difference") else 970),
            edgecolors="#64748b", linewidths=0.9, node_shape="o")
    nx.draw_networkx_edges(
        g, positions, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12,
        node_size=1220, min_source_margin=8, min_target_margin=8,
        edge_color="#94a3b8", width=1.2, alpha=0.9,
        connectionstyle="arc3,rad=0.03")
    nx.draw_networkx_labels(
        g, positions, labels={n: d["label"] for n, d in g.nodes(data=True)},
        font_size=9, ax=ax)
    # Знаки и деление на соответствующую массу — часть вычисления ускорений.
    signs = {(a, b): attr["label"] for a, b, attr in g.edges(data=True)
             if "label" in attr}
    if signs:
        nx.draw_networkx_edge_labels(g, positions, edge_labels=signs,
                                     font_size=7, rotate=False, ax=ax,
                                     label_pos=0.67,
                                     bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.85})
    ax.set_xlim(-0.95, 9.0)
    ax.set_ylim(0.15, 8.65)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=5)
    return fig, ax
