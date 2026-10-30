"""Análisis completo del experimento.   Uso: python analysis.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from abtest.data import make_experiment  # noqa: E402
from abtest.stats import bootstrap_rel_ci, cuped, holm, sample_size_per_arm, srm_pvalue, welch  # noqa: E402

ROOT = Path(__file__).parent
FIG = ROOT / "reports" / "figures"
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e0"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "legend.frameon": False, "font.size": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = make_experiment()
    df.to_csv(ROOT / "data" / "experiment_synthetic.csv", index=False)
    c, t = df[df.group == "control"], df[df.group == "tratamiento"]
    res: dict = {}

    # 1. Diseño: tamaño de muestra para un MDE de 5 % sobre el ingreso pre-periodo
    sd, mean = df.revenue_pre.std(), df.revenue_pre.mean()
    res["n_requerido_por_grupo_MDE5pct_sin_CUPED"] = sample_size_per_arm(sd, 0.05 * mean)
    rho = np.corrcoef(df.revenue_pre, df.revenue)[0, 1]
    res["n_requerido_por_grupo_MDE5pct_con_CUPED"] = sample_size_per_arm(sd * np.sqrt(1 - rho**2), 0.05 * mean)
    res["n_real_por_grupo"] = int(len(c))

    # 2. Calidad del experimento
    res["SRM_p_value"] = round(srm_pvalue(len(c), len(t)), 4)
    pre = welch(c.revenue_pre, t.revenue_pre)
    res["balance_pre_periodo_p_value"] = round(pre.p_value, 4)

    # 3. Efecto principal: sin y con CUPED
    raw = welch(c.revenue, t.revenue)
    y_adj, theta = cuped(df.revenue.to_numpy(), df.revenue_pre.to_numpy())
    df["revenue_cuped"] = y_adj
    adj = welch(df.loc[df.group == "control", "revenue_cuped"], df.loc[df.group == "tratamiento", "revenue_cuped"])
    boot = bootstrap_rel_ci(c.revenue, t.revenue)
    var_red = 1 - df.revenue_cuped.var() / df.revenue.var()
    res["efecto_sin_CUPED"] = {"lift_rel": round(raw.rel, 4), "IC95_abs": [round(raw.ci_low, 1), round(raw.ci_high, 1)],
                               "p_value": round(raw.p_value, 4), "IC95_bootstrap_rel": [round(b, 4) for b in boot]}
    res["efecto_con_CUPED"] = {"lift_rel": round(adj.diff / c.revenue.mean(), 4),
                               "IC95_rel": [round(adj.ci_low / c.revenue.mean(), 4), round(adj.ci_high / c.revenue.mean(), 4)],
                               "p_value": round(adj.p_value, 6), "theta": round(theta, 3),
                               "reduccion_varianza": round(var_red, 3)}

    # 4. Guardrail: margen
    g = welch(c.margin_pct, t.margin_pct)
    res["guardrail_margen_pp"] = {"diff_pp": round(g.diff * 100, 3), "IC95_pp": [round(g.ci_low * 100, 3), round(g.ci_high * 100, 3)],
                                  "p_value": round(g.p_value, 4)}

    # 5. Heterogeneidad por sector (exploratorio, con corrección de Holm)
    sec = {}
    for s, d in df.groupby("sector"):
        e = welch(d.loc[d.group == "control", "revenue_cuped"], d.loc[d.group == "tratamiento", "revenue_cuped"])
        m = d.loc[d.group == "control", "revenue"].mean()
        sec[s] = {"lift_rel": e.diff / m, "lo": e.ci_low / m, "hi": e.ci_high / m, "p": e.p_value, "n": len(d)}
    adj_p = holm({s: v["p"] for s, v in sec.items()})
    res["por_sector_CUPED"] = {s: {"n": v["n"], "lift_rel": round(v["lift_rel"], 4),
                                   "IC95_rel": [round(v["lo"], 4), round(v["hi"], 4)],
                                   "p": round(v["p"], 4), "p_holm": round(adj_p[s], 4)} for s, v in sec.items()}

    (ROOT / "reports" / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # Figura: forest plot (efecto relativo con IC95)
    rows = [("Total sin CUPED", raw.rel, raw.ci_low / c.revenue.mean(), raw.ci_high / c.revenue.mean())]
    rows.append(("Total con CUPED", *[res["efecto_con_CUPED"]["lift_rel"]], *res["efecto_con_CUPED"]["IC95_rel"]))
    rows += [(f"{s} (CUPED)", v["lift_rel"], v["lo"], v["hi"]) for s, v in sec.items()]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (lab, est, lo, hi) in enumerate(rows[::-1]):
        col = C2 if lab.startswith("Total") else C1
        ax.plot([lo * 100, hi * 100], [i, i], color=col, linewidth=2)
        ax.plot(est * 100, i, "o", color=col, markersize=7)
        ax.text(hi * 100 + 0.4, i, f"{est*100:+.1f}%  [{lo*100:+.1f}, {hi*100:+.1f}]", va="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("Efecto en ingreso por cliente (%), IC 95 %")
    ax.set_title("Impacto de las recomendaciones de venta cruzada")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(r[2] for r in rows) * 100 - 2, max(r[3] for r in rows) * 100 + 9)
    fig.tight_layout(); fig.savefig(FIG / "01_forest_plot.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
