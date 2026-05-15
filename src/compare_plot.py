"""
학습 데이터 전체 + 재학습 전후 비교 시각화

실행: python3 src/compare_plot.py
출력: data/plot_overview.png      — 2년치 전체 학습 데이터 + 이상치 구간
      data/plot_seasonal.png      — 월별/시간대별 평균 패턴
      data/plot_compare_24h.png   — 재학습 전후 24시간 예측 비교
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

plt.rcParams["font.family"] = "DejaVu Sans"

MIN_PODS = 1
MAX_PODS = 8


def load(path: Path, label: str = "", parse_ds: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if label:
            print(f"[경고] {path.name} 없음 — {label} 생략")
        return None
    kw = {"parse_dates": ["ds"]} if parse_ds else {}
    df = pd.read_csv(path, **kw)
    print(f"[로드] {path.name}: {len(df):,}행")
    return df


# ── Plot 1: 2년치 전체 데이터 개요 ───────────────────────────

def plot_overview(train: pd.DataFrame, anomalies: pd.DataFrame | None):
    # sharex 제거 — 하단 막대 차트가 정수 인덱스라 날짜 축이 눌리는 문제 방지
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle("Training Data Overview (2023–2024)", fontsize=15, fontweight="bold")

    daily = train.set_index("ds").resample("D").mean(numeric_only=True).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    # 상단: RPS 일별 평균
    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["y"], color="steelblue", linewidth=1.2, label="Daily avg RPS")
    ax1.set_ylabel("Request Rate (req/min)")
    ax1.set_title("Daily Average Request Rate")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 이상치 구간 음영
    if anomalies is not None:
        colors = {1: "red", 2: "orange", 3: "purple"}
        labels = {1: "Scenario1: Planting surge", 2: "Scenario2: Typhoon surge", 3: "Scenario3: Monsoon irregular"}
        added = set()
        for _, row in anomalies.iterrows():
            s, e, sc = pd.Timestamp(row["start"]), pd.Timestamp(row["end"]), int(row["scenario"])
            c = colors.get(sc, "gray")
            lbl = labels.get(sc) if sc not in added else None
            ax1.axvspan(s, e, alpha=0.25, color=c, label=lbl)
            added.add(sc)
        ax1.legend(fontsize=8, loc="upper right")

    # 중단: 장마/태풍 regressor
    ax2 = axes[1]
    ax2.fill_between(daily["ds"], daily["is_monsoon"], alpha=0.5, color="skyblue", label="Monsoon (is_monsoon)")
    ax2.fill_between(daily["ds"], daily["typhoon_index"], alpha=0.6, color="coral", label="Typhoon index")
    ax2.set_ylabel("Regressor value")
    ax2.set_title("Weather Regressors")
    ax2.set_ylim(-0.05, 1.1)
    ax2.xaxis.set_major_locator(date_loc)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 하단: 월별 평균 막대 (독립 x축)
    ax3 = axes[2]
    monthly_avg = train.groupby(train["ds"].dt.to_period("M"))["y"].mean()
    x_pos = range(len(monthly_avg))
    ax3.bar(x_pos, monthly_avg.values, color="steelblue", alpha=0.7, width=0.8)
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels([str(m) for m in monthly_avg.index], rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Avg RPS (req/min)")
    ax3.set_title("Monthly Average Request Rate")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "plot_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 2: 계절 패턴 (월별 / 시간대별) ─────────────────────

def plot_seasonal(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Seasonal Patterns in Training Data", fontsize=13, fontweight="bold")

    # 월별 평균
    ax1 = axes[0]
    monthly = train.groupby(train["ds"].dt.month)["y"].mean()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    colors = ["tomato" if v == monthly.max() else "steelblue" for v in monthly.values]
    ax1.bar(monthly.index, monthly.values, color=colors, alpha=0.8, width=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Avg RPS (req/min)")
    ax1.set_title("Monthly Seasonality\n(red = peak)")
    ax1.grid(True, alpha=0.3, axis="y")

    # 시간대별 평균
    ax2 = axes[1]
    hourly = train.groupby(train["ds"].dt.hour)["y"].mean()
    colors2 = ["tomato" if v >= hourly.quantile(0.85) else "steelblue" for v in hourly.values]
    ax2.bar(hourly.index, hourly.values, color=colors2, alpha=0.8, width=0.7)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Avg RPS (req/min)")
    ax2.set_title("Daily Seasonality\n(red = peak hours)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "plot_seasonal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 3: 재학습 전후 24시간 예측 비교 ─────────────────────

def plot_compare_24h(old: pd.DataFrame | None, new: pd.DataFrame | None):
    hours = list(range(1, 25))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Prophet Autoscaling — Before vs After Retrain (next 24h)", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.set_title("Predicted RPS")
    ax1.set_ylabel("Request Rate (req/min)")
    ax1.set_xlabel("Hour (T+1 ~ T+24)")

    if old is not None:
        x = hours[:len(old)]
        ax1.plot(x, old["yhat"].values, label=f"Before retrain  ({old['ds'].iloc[0].strftime('%Y-%m-%d')} ~)",
                 color="steelblue", linewidth=2, marker="o", markersize=4)
        if "yhat_lower" in old.columns:
            ax1.fill_between(x, old["yhat_lower"].values, old["yhat_upper"].values,
                             alpha=0.15, color="steelblue")

    if new is not None:
        x = hours[:len(new)]
        ax1.plot(x, new["yhat"].values, label=f"After retrain   ({new['ds'].iloc[0].strftime('%Y-%m-%d')} ~)",
                 color="tomato", linewidth=2, marker="s", markersize=4, linestyle="--")
        if "yhat_lower" in new.columns:
            ax1.fill_between(x, new["yhat_lower"].values, new["yhat_upper"].values,
                             alpha=0.15, color="tomato")

    ax1.set_xticks(hours)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.set_title("Required Pod Count")
    ax2.set_ylabel("Pod Count")
    ax2.set_xlabel("Hour (T+1 ~ T+24)")

    if old is not None and "required_pods" in old.columns:
        ax2.step(hours[:len(old)], old["required_pods"].values, label="Before retrain",
                 color="steelblue", linewidth=2, where="post")

    if new is not None and "required_pods" in new.columns:
        ax2.step(hours[:len(new)], new["required_pods"].values, label="After retrain",
                 color="tomato", linewidth=2, linestyle="--", where="post")

    ax2.set_xticks(hours)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "plot_compare_24h.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── Plot 4: 연간 Pod 수 추이 ──────────────────────────────────

def plot_annual_pods(annual: pd.DataFrame):
    daily = (
        annual.set_index("ds")
        .resample("D")
        .agg({"yhat": "mean", "yhat_lower": "mean", "yhat_upper": "mean", "required_pods": "max"})
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("Annual Pod Scaling Forecast (2025)", fontsize=15, fontweight="bold")

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    # 상단: 예측 RPS
    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["yhat"], color="steelblue", linewidth=1.2, label="Daily avg RPS")
    ax1.fill_between(daily["ds"], daily["yhat_lower"], daily["yhat_upper"], alpha=0.2, color="steelblue")
    ax1.axvspan(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31"), alpha=0.15, color="red",   label="March Peak (planting)")
    ax1.axvspan(pd.Timestamp("2025-06-25"), pd.Timestamp("2025-07-25"), alpha=0.15, color="skyblue", label="Monsoon")
    ax1.axvspan(pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-30"), alpha=0.10, color="orange", label="Typhoon season")
    ax1.set_ylabel("Predicted RPS (req/min)")
    ax1.set_title("Predicted Request Rate")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 하단: 필요 Pod 수
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
    out = DATA_DIR / "plot_annual_pods.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    train      = load(DATA_DIR / "dummy_request_rate.csv",          "학습 데이터")
    anomalies  = load(DATA_DIR / "dummy_anomaly_events.csv", parse_ds=False)
    old_pred   = load(DATA_DIR / "predictions_request_rate.csv",    "재학습 전 예측")
    new_pred   = load(DATA_DIR / "predictions_retrained.csv",       "재학습 후 예측")
    annual     = load(DATA_DIR / "predictions_annual_2025.csv",     "연간 예측")

    print()
    if train is not None:
        plot_overview(train, anomalies)
        plot_seasonal(train)

    if old_pred is not None or new_pred is not None:
        plot_compare_24h(old_pred, new_pred)

    if annual is not None:
        plot_annual_pods(annual)

    print("\n생성된 파일:")
    print("  data/plot_overview.png    — 2년치 전체 + 이상치 구간")
    print("  data/plot_seasonal.png    — 월별/시간대별 계절 패턴")
    print("  data/plot_compare_24h.png — 재학습 전후 24h 예측 비교")
    print("  data/plot_annual_pods.png — 2025년 연간 Pod 수 추이")


if __name__ == "__main__":
    main()
