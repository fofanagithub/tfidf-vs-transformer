"""Build the final comparison report (Markdown + figures + JSON) from saved predictions.

The report is computed only from files written by the training runs, so it can be regenerated
at any time without retraining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import disagreements, slice_metrics  # noqa: E402
from .baseline import load_predictions  # noqa: E402
from .config import ExperimentConfig  # noqa: E402
from .data import Splits  # noqa: E402
from .evaluation import evaluate, mcnemar_test, paired_bootstrap_diff  # noqa: E402
from .utils import get_logger, load_json, save_json  # noqa: E402

log = get_logger()

# Categorical slots 1-2 of the validated reference palette (CVD-safe pair), sequential blue ramp.
COLORS = {"baseline": "#2a78d6", "transformer": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def _load_transformer_runs(run_dir: Path) -> list[dict]:
    runs = []
    for seed_dir in sorted((run_dir / "transformer").glob("seed_*")):
        if (seed_dir / "predictions.csv").exists():
            y, pred, probs = load_predictions(seed_dir / "predictions.csv")
            runs.append({"meta": load_json(seed_dir / "meta.json"), "y": y, "pred": pred, "probs": probs})
    return runs


def _fmt_ci(value: float, ci: list[float]) -> str:
    return f"{100 * value:.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]"


def _fmt_p(p: float, n_resamples: int) -> str:
    return f"< {1 / n_resamples:.3g}" if p <= 1 / n_resamples else f"{p:.3g}"


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _plot_confusions(cms: dict[str, list], label_names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.6))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        cm = np.asarray(cm, dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{100 * norm[i, j]:.1f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(label_names)), label_names, rotation=30, ha="right")
        ax.set_yticks(range(len(label_names)), label_names)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(name, color=INK)
    fig.savefig(path)
    plt.close(fig)


def _plot_reliability(probs: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Calibration parfaite")
    bins = np.linspace(0, 1, n_bins + 1)
    for name, (y, p) in probs.items():
        conf, correct = p.max(axis=1), (p.argmax(axis=1) == y)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
        xs = [conf[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        ys = [correct[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", ms=5, lw=2, color=COLORS[name], label=name)
    ax.set_xlabel("Confiance moyenne")
    ax.set_ylabel("Exactitude observée")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def _plot_learning_curve(lc: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    agg = lc.groupby(["model", "train_size"])["test_macro_f1"].agg(["mean", "std"]).reset_index()
    for model, grp in agg.groupby("model"):
        grp = grp.sort_values("train_size")
        std = grp["std"].fillna(0)
        ax.fill_between(grp["train_size"], 100 * (grp["mean"] - std), 100 * (grp["mean"] + std), color=COLORS[model], alpha=0.15, lw=0)
        ax.plot(grp["train_size"], 100 * grp["mean"], marker="o", ms=5, lw=2, color=COLORS[model], label=model)
        last = grp.iloc[-1]
        ax.annotate(f"{100 * last['mean']:.1f}", (last["train_size"], 100 * last["mean"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_xscale("log")
    sizes = sorted(lc["train_size"].unique())
    ax.set_xticks(sizes, [f"{s:,}".replace(",", " ") for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Exemples d'entraînement (échelle log)")
    ax.set_ylabel("Macro-F1 test (%)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def _plot_slices(slices: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(slices))
    width = 0.38
    for k, model in enumerate(("baseline", "transformer")):
        ax.bar(x + (k - 0.5) * width, 100 * slices[f"{model}_accuracy"], width - 0.02, color=COLORS[model], label=model)
    ticks = [s.replace(" mots", "\nmots").replace(" négation", "\nnégation") + f"\n(n={n})" for s, n in zip(slices["slice"], slices["n"])]
    ax.set_xticks(x, ticks, fontsize=7.5)
    lo = 100 * min(slices["baseline_accuracy"].min(), slices["transformer_accuracy"].min())
    ax.set_ylim(max(0, lo - 5), 100)
    ax.set_ylabel("Exactitude (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def build_report(cfg: ExperimentConfig, splits: Splits) -> dict:
    run_dir = cfg.run_dir
    out = run_dir / "report"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label_names = splits.label_names
    test = splits.test

    y, base_pred, base_probs = load_predictions(run_dir / "baseline" / "predictions.csv")
    base_meta = load_json(run_dir / "baseline" / "meta.json")
    runs = _load_transformer_runs(run_dir)
    if not runs:
        raise FileNotFoundError(f"No transformer run found in {run_dir / 'transformer'}")
    for r in [{"y": y}, *runs]:
        if len(r["y"]) != len(test) or not np.array_equal(r["y"], test["label"].to_numpy()):
            raise ValueError("Predictions do not match the current test split: rerun the models with the same config.")

    # --- Metrics ---------------------------------------------------------------------------------
    base_eval = evaluate(y, base_pred, base_probs, label_names, cfg.n_bootstrap)
    for r in runs:
        r["eval"] = evaluate(y, r["pred"], r["probs"], label_names, cfg.n_bootstrap)
        r["mcnemar"] = mcnemar_test(y, base_pred, r["pred"])
        r["bootstrap_diff"] = paired_bootstrap_diff(y, base_pred, r["pred"], n_resamples=cfg.n_bootstrap)

    def agg(key: str) -> tuple[float, float]:
        vals = np.array([r["eval"][key] for r in runs])
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    # Representative run for figures / error analysis: median *validation* score (the test set is not used to choose).
    rep = sorted(runs, key=lambda r: r["meta"]["val_macro_f1"])[len(runs) // 2]

    # --- Tables -----------------------------------------------------------------------------------
    perf_rows = [
        {
            "Modèle": f"TF-IDF + LogReg ({base_meta['best_params']['feature_set']}, C={base_meta['best_params']['C']:g})",
            "Accuracy (IC 95 %)": _fmt_ci(base_eval["accuracy"], base_eval["accuracy_ci95"]),
            "Macro-F1 (IC 95 %)": _fmt_ci(base_eval["macro_f1"], base_eval["macro_f1_ci95"]),
            "ECE": f"{100 * base_eval['ece']:.2f}",
        }
    ]
    for r in runs:
        perf_rows.append(
            {
                "Modèle": f"{r['meta']['model']} (seed {r['meta']['seed']})",
                "Accuracy (IC 95 %)": _fmt_ci(r["eval"]["accuracy"], r["eval"]["accuracy_ci95"]),
                "Macro-F1 (IC 95 %)": _fmt_ci(r["eval"]["macro_f1"], r["eval"]["macro_f1_ci95"]),
                "ECE": f"{100 * r['eval']['ece']:.2f}",
            }
        )
    (acc_m, acc_s), (f1_m, f1_s), (ece_m, ece_s) = agg("accuracy"), agg("macro_f1"), agg("ece")
    perf_rows.append(
        {
            "Modèle": f"**{runs[0]['meta']['model']} — moyenne ± écart-type ({len(runs)} seeds)**",
            "Accuracy (IC 95 %)": f"**{100 * acc_m:.2f} ± {100 * acc_s:.2f}**",
            "Macro-F1 (IC 95 %)": f"**{100 * f1_m:.2f} ± {100 * f1_s:.2f}**",
            "ECE": f"**{100 * ece_m:.2f} ± {100 * ece_s:.2f}**",
        }
    )
    perf = pd.DataFrame(perf_rows)

    sig = pd.DataFrame(
        [
            {
                "Seed": r["meta"]["seed"],
                "Δ Macro-F1 (pts)": f"{100 * r['bootstrap_diff']['diff']:+.2f} [{100 * r['bootstrap_diff']['ci95'][0]:+.2f}, {100 * r['bootstrap_diff']['ci95'][1]:+.2f}]",
                "p (bootstrap apparié)": _fmt_p(r["bootstrap_diff"]["p_value"], cfg.n_bootstrap),
                "Seule baseline correcte": r["mcnemar"]["a_only_correct"],
                "Seul transformer correct": r["mcnemar"]["b_only_correct"],
                "p (McNemar)": f"{r['mcnemar']['p_value']:.3g}",
            }
            for r in runs
        ]
    )

    t_train = np.mean([r["meta"]["train_seconds"] for r in runs])
    cost = pd.DataFrame(
        [
            {
                "Modèle": "TF-IDF + LogReg",
                "Paramètres": f"{base_meta['n_parameters']:,}".replace(",", " "),
                "Taille (Mo)": f"{base_meta['model_size_mb']:.1f}",
                "Entraînement": f"{base_meta['train_seconds']:.0f} s",
                "Inférence (ex./s)": f"{base_meta['inference_examples_per_second']:.0f}",
                "Matériel d'inférence": base_meta["inference_device"],
            },
            {
                "Modèle": runs[0]["meta"]["model"],
                "Paramètres": f"{runs[0]['meta']['n_parameters']:,}".replace(",", " "),
                "Taille (Mo)": f"{runs[0]['meta']['model_size_mb']:.1f}",
                "Entraînement": f"{t_train / 60:.1f} min",
                "Inférence (ex./s)": f"{np.mean([r['meta']['inference_examples_per_second'] for r in runs]):.0f}",
                "Matériel d'inférence": runs[0]["meta"]["inference_device"],
            },
        ]
    )

    # --- Error analysis -------------------------------------------------------------------------
    slices = slice_metrics(test, {"baseline": base_pred, "transformer": rep["pred"]}, cfg.data.language)
    slices.round(4).to_csv(out / "slices.csv", index=False)
    disagreements(test, label_names, base_pred, base_probs, rep["pred"], rep["probs"]).to_csv(out / "disagreements.csv", index=False)

    # --- Figures --------------------------------------------------------------------------------
    _plot_confusions({"TF-IDF + LogReg": base_eval["confusion_matrix"], f"{rep['meta']['model']} (seed {rep['meta']['seed']})": rep["eval"]["confusion_matrix"]},
                     label_names, fig_dir / "confusion_matrices.png")
    _plot_reliability({"baseline": (y, base_probs), "transformer": (y, rep["probs"])}, fig_dir / "reliability.png")
    _plot_slices(slices, fig_dir / "slices.png")
    lc_path = run_dir / "learning_curve" / "learning_curve.csv"
    lc_section = "_Courbe d'apprentissage non calculée (`tvt learning-curve`)._"
    if lc_path.exists():
        lc = pd.read_csv(lc_path)
        if set(lc["model"]) >= {"baseline", "transformer"}:
            _plot_learning_curve(lc, fig_dir / "learning_curve.png")
            stats = lc.groupby(["train_size", "model"])["test_macro_f1"].agg(["mean", "std"]).fillna(0.0)
            cells = (100 * stats["mean"]).map("{:.2f}".format) + " ± " + (100 * stats["std"]).map("{:.2f}".format)
            lc_table = cells.unstack("model")[["baseline", "transformer"]].rename_axis(None, axis=1).reset_index()
            lc_table = lc_table.rename(columns={"train_size": "Exemples"})
            lc_section = f"![Courbe d'apprentissage](figures/learning_curve.png)\n\nMacro-F1 test (%), moyenne ± écart-type sur {lc['seed'].nunique()} tirage(s) du sous-échantillon :\n\n{_md_table(lc_table)}"

    # --- Markdown -------------------------------------------------------------------------------
    data_info = splits.summary()
    slice_tbl = slices.assign(**{c: (100 * slices[c]).round(2) for c in slices.columns if c.endswith(("accuracy", "macro_f1"))})
    slice_tbl = slice_tbl.rename(columns={
        "slice": "Tranche", "baseline_accuracy": "Acc. baseline", "baseline_macro_f1": "F1 baseline",
        "transformer_accuracy": "Acc. transformer", "transformer_macro_f1": "F1 transformer",
    })
    md = f"""# Rapport — {cfg.name}

Jeu de données : `{cfg.data.name}` · classes : {", ".join(label_names)} ·
train / val / test = {len(splits.train)} / {len(splits.val)} / {len(test)} · empreinte du test : `{data_info['test_fingerprint']}`

## 1. Performance sur le test

{_md_table(perf)}

Les intervalles de confiance à 95 % sont obtenus par bootstrap percentile ({cfg.n_bootstrap} rééchantillonnages du test).
ECE = erreur de calibration attendue (15 intervalles, en points de %).

## 2. Significativité de l'écart (transformer − baseline)

{_md_table(sig)}

## 3. Coût

{_md_table(cost)}

## 4. Efficacité en données

{lc_section}

## 5. Analyse d'erreurs (transformer : seed {rep['meta']['seed']}, choisi par score de validation médian)

![Matrices de confusion](figures/confusion_matrices.png)

![Exactitude par tranche](figures/slices.png)

{_md_table(slice_tbl)}

![Calibration](figures/reliability.png)

Les exemples de désaccord (erreurs les plus confiantes de chaque modèle) sont dans `disagreements.csv`.
"""
    (out / "report.md").write_text(md, encoding="utf-8")

    summary = {
        "baseline": {"meta": base_meta, "eval": base_eval},
        "transformer": {
            "runs": [{"meta": r["meta"], "eval": r["eval"], "mcnemar": r["mcnemar"], "bootstrap_diff": r["bootstrap_diff"]} for r in runs],
            "mean": {"accuracy": acc_m, "macro_f1": f1_m, "ece": ece_m},
            "std": {"accuracy": acc_s, "macro_f1": f1_s, "ece": ece_s},
        },
        "data": data_info,
    }
    save_json(summary, out / "summary.json")
    log.info("Report written to %s", out / "report.md")
    return summary
