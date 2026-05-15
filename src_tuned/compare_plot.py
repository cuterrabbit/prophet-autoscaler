"""
튜닝 버전: baseline / log 변환 / tuned 3-way 시각화 비교

실행: python src_tuned/compare_plot.py
출력: data/plot_tuned_overview.png        — 학습 데이터 개요
      data/plot_tuned_annual_pods.png     — tuned 모델 연간 Pod 추이
      data/plot_tuned_3way_pods.png       — 3버전 Pod 비교
      data/plot_tuned_hp_search.png       — 하이퍼파라미터 탐색 결과 히트맵
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MIN_PODS = 1
MAX_PODS = 8

plt.rcParams["font.family"] = "DejaVu Sans"


def load(path: Path, label: str = "", parse_ds: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if label:
            print(f"[경고] {path.name} 없음 — {label} 생략")
        return None
    kw = {"parse_dates": ["ds"]} if parse_ds else {}
    df = pd.read_csv(path, **kw)
    print(f"[로드] {path.name}: {len(df):,}행")
    return df


# ── Plot 1: 학습 데이터 개요 ──────────────────────────────────

def plot_overview(train: pd.DataFrame, anomalies: pd.DataFrame | None):
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    fig.suptitle("Training Data Overview (2023–2024)", fontsize=15, fontweight="bold")

    daily    = train.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["y"], color="steelblue", linewidth=1.2)
    ax1.set_ylabel("Request Rate (req/min)")
    ax1.set_title("Daily Average Request Rate")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if anomalies is not None:
        colors = {1: "red", 2: "orange", 3: "purple"}
        labels = {1: "Scenario1: Planting surge", 2: "Scenario2: Typhoon surge", 3: "Scenario3: Monsoon irregular"}
        added  = set()
        for _, row in anomalies.iterrows():
            s, e, sc = pd.Timestamp(row["start"]), pd.Timestamp(row["end"]), int(row["scenario"])
            lbl = labels.get(sc) if sc not in added else None
            ax1.axvspan(s, e, alpha=0.25, color=colors.get(sc, "gray"), label=lbl)
            added.add(sc)
        ax1.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    monthly_avg = train.groupby(train["ds"].dt.to_period("M"))["y"].mean()
    x_pos = range(len(monthly_avg))
    ax2.bar(x_pos, monthly_avg.values, color="steelblue", alpha=0.7, width=0.8)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([str(m) for m in monthly_avg.index], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Avg RPS (req/min)")
    ax2.set_title("Monthly Average Request Rate")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "plot_tuned_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 2: tuned 모델 연간 Pod 추이 ──────────────────────────

def plot_annual_pods(annual: pd.DataFrame):
    daily = (
        annual.set_index("ds")
        .resample("D")
        .agg({"yhat": "mean", "yhat_lower": "mean", "yhat_upper": "mean", "required_pods": "max"})
        .reset_index()
    )

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("Tuned Annual Pod Scaling Forecast — 2025 (log + HP tuned)", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["yhat"], color="steelblue", linewidth=1.2, label="Daily avg RPS")
    ax1.fill_between(daily["ds"], daily["yhat_lower"], daily["yhat_upper"], alpha=0.2, color="steelblue")
    ax1.axvspan(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31"), alpha=0.15, color="red",     label="March Peak")
    ax1.axvspan(pd.Timestamp("2025-06-25"), pd.Timestamp("2025-07-25"), alpha=0.15, color="skyblue", label="Monsoon")
    ax1.axvspan(pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-30"), alpha=0.10, color="orange",  label="Typhoon season")
    ax1.set_ylabel("Predicted RPS (req/min)")
    ax1.set_title("Predicted Request Rate")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.step(daily["ds"], daily["required_pods"], color="tomato", linewidth=1.5, where="post", label="Required Pods (daily max)")
    ax2.fill_between(daily["ds"], MIN_PODS, daily["required_pods"], alpha=0.25, color="tomato", step="post")
    ax2.axhline(y=MAX_PODS, color="gray",  linestyle="--", linewidth=1, label=f"MAX_PODS = {MAX_PODS}")
    ax2.axhline(y=MIN_PODS, color="green", linestyle="--", linewidth=1, label=f"MIN_PODS = {MIN_PODS}")
    ax2.axvspan(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31"), alpha=0.15, color="red")
    ax2.set_ylim(0, MAX_PODS + 1)
    ax2.set_ylabel("Pod Count")
    ax2.set_title(f"Required Pod Count  (MIN={MIN_PODS} ~ MAX={MAX_PODS})")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(date_loc)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "plot_tuned_annual_pods.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 3: 3-way Pod 비교 ─────────────────────────────────────

def plot_3way_pods(baseline: pd.DataFrame, optimized: pd.DataFrame, tuned: pd.DataFrame):
    b_d = baseline.set_index("ds").resample("D")["required_pods"].max().reset_index()
    o_d = optimized.set_index("ds").resample("D")["required_pods"].max().reset_index()
    t_d = tuned.set_index("ds").resample("D")["required_pods"].max().reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("3-Way Pod Comparison — Baseline / Log Transform / Tuned (2025)",
                 fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.step(b_d["ds"], b_d["required_pods"], color="steelblue", linewidth=1.5, where="post", label="Baseline")
    ax1.step(o_d["ds"], o_d["required_pods"], color="tomato",    linewidth=1.5, where="post", label="Log transform",  linestyle="--")
    ax1.step(t_d["ds"], t_d["required_pods"], color="seagreen",  linewidth=1.5, where="post", label="Tuned (HP opt)", linestyle=":")
    ax1.axhline(y=MAX_PODS, color="gray",  linestyle=":", linewidth=1, label=f"MAX={MAX_PODS}")
    ax1.axhline(y=MIN_PODS, color="gray",  linestyle=":", linewidth=1, label=f"MIN={MIN_PODS}")
    ax1.set_ylim(0, MAX_PODS + 1)
    ax1.set_ylabel("Pod Count (daily max)")
    ax1.set_title("Pod Count — Baseline vs Log Transform vs Tuned")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff = t_d["required_pods"].values - b_d["required_pods"].values
    colors = ["tomato" if d > 0 else ("seagreen" if d < 0 else "gray") for d in diff]
    ax2.bar(t_d["ds"], diff, color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_ylabel("Pod Diff (tuned - baseline)")
    ax2.set_title("Pod Count Difference  (red = tuned 더 많음 / green = 더 적음)")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(date_loc)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "plot_tuned_3way_pods.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 4: 하이퍼파라미터 탐색 결과 히트맵 ──────────────────

def plot_hp_heatmap(hp: pd.DataFrame):
    pivot = hp.pivot(index="seasonality_prior_scale", columns="changepoint_prior_scale", values="smape")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Hyperparameter Search — SMAPE Heatmap", fontsize=13, fontweight="bold")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="SMAPE (lower = better)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("changepoint_prior_scale")
    ax.set_ylabel("seasonality_prior_scale")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if val > pivot.values.mean() else "black")

    best_idx = np.unravel_index(np.nanargmin(pivot.values), pivot.values.shape)
    ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=2.5, label="Best"))
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = DATA_DIR / "plot_tuned_hp_search.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    train      = load(DATA_DIR / "dummy_request_rate.csv",           "학습 데이터")
    anomalies  = load(DATA_DIR / "dummy_anomaly_events.csv",         parse_ds=False)
    annual_bl  = load(DATA_DIR / "predictions_annual_2025.csv",      "baseline 연간")
    annual_opt = load(DATA_DIR / "predictions_log_annual_2025.csv",  "log 연간")
    annual_tnd = load(DATA_DIR / "predictions_tuned_annual_2025.csv","tuned 연간")
    hp_results = load(DATA_DIR / "hp_search_results.csv",            "HP 탐색 결과", parse_ds=False)

    print()
    if train is not None:
        plot_overview(train, anomalies)

    if annual_tnd is not None:
        plot_annual_pods(annual_tnd)

    if annual_bl is not None and annual_opt is not None and annual_tnd is not None:
        plot_3way_pods(annual_bl, annual_opt, annual_tnd)

    if hp_results is not None:
        plot_hp_heatmap(hp_results)

    print("\n생성된 파일:")
    print("  data/plot_tuned_overview.png      — 학습 데이터 개요")
    print("  data/plot_tuned_annual_pods.png   — tuned 모델 연간 Pod 추이")
    print("  data/plot_tuned_3way_pods.png     — 3버전 Pod 비교")
    print("  data/plot_tuned_hp_search.png     — HP 탐색 결과 히트맵")


if __name__ == "__main__":
    main()
