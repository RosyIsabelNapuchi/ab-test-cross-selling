"""Experimento SINTÉTICO: recomendaciones de venta cruzada para vendedores B2B.

Unidad de aleatorización: cliente. Métrica principal: ingreso por cliente en 8
semanas. Se simula un efecto real de +5 % en ingreso (más fuerte en Industria) y
un leve costo en margen, para mostrar el uso de una métrica de guardrail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_experiment(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sector = rng.choice(["Pesca", "Industria", "Mineria"], n, p=[0.3, 0.45, 0.25])
    group = rng.permutation(np.repeat(["control", "tratamiento"], n // 2))
    base = rng.lognormal(9.0, 0.9, n)                          # nivel de compra del cliente
    pre = base * rng.lognormal(0, 0.35, n)                       # ingreso 8 semanas previas
    lift = np.where(group == "tratamiento", np.select([sector == "Industria", sector == "Pesca"], [0.08, 0.03], 0.02), 0)
    post = base * (1 + lift) * rng.lognormal(0, 0.35, n)
    margin = np.clip(rng.normal(0.27, 0.05, n) - np.where(group == "tratamiento", 0.003, 0), 0.05, 0.6)
    return pd.DataFrame({"customer_id": [f"K{i:05d}" for i in range(n)], "sector": sector, "group": group,
                         "revenue_pre": pre.round(2), "revenue": post.round(2), "margin_pct": margin.round(4)})
