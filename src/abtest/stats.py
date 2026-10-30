"""Herramientas estadísticas para experimentos A/B, escritas con numpy/scipy para que sean auditables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Effect:
    diff: float          # tratamiento - control (unidades de la métrica)
    rel: float           # diferencia relativa vs. control
    ci_low: float
    ci_high: float
    p_value: float


def sample_size_per_arm(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """n por grupo para detectar una diferencia absoluta `mde_abs` (test bilateral, varianzas iguales)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb) ** 2 * sd**2 / mde_abs**2))


def srm_pvalue(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> float:
    """Sample Ratio Mismatch: chi-cuadrado contra la asignación esperada."""
    total = n_control + n_treatment
    exp = [total * (1 - expected_ratio), total * expected_ratio]
    return float(stats.chisquare([n_control, n_treatment], exp).pvalue)


def welch(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> Effect:
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    diff = t.mean() - c.mean()
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(vc + vt)
    dof = (vc + vt) ** 2 / (vc**2 / (len(c) - 1) + vt**2 / (len(t) - 1))
    q = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff / se), dof)
    return Effect(diff, diff / c.mean(), diff - q * se, diff + q * se, float(p))


def bootstrap_rel_ci(control, treatment, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """IC percentil del efecto relativo (robusto a colas largas de ingresos)."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    mc = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1)
    mt = t[rng.integers(0, len(t), (n_boot, len(t)))].mean(1)
    rel = mt / mc - 1
    return float(np.quantile(rel, alpha / 2)), float(np.quantile(rel, 1 - alpha / 2))


def cuped(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: y_adj = y - theta (x - mean(x)), theta = cov(y,x)/var(x). Reduce varianza sin sesgar el efecto."""
    y, x = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Ajuste de Holm-Bonferroni para comparaciones múltiples (p ajustados)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
