"""
모델 검증
참조: module-spec/module-spec-1-2.md

실행: python src/validate.py
출력: data/plot_validation_fit.png      — 전체 기간 실제 vs 예측
      data/plot_validation_holdout.png  — 홀드아웃(2024) 실제 vs 예측
"""
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CAPACITY_PER_POD = 21.1
SAFETY_MARGIN    = 0.2
MIN_PODS = 1
MAX_PODS = 8

plt.rcParams["font.family"] = "DejaVu Sans"


# ── 공통 ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATA_DIR / "dummy_request_rate.csv"
    df = pd.read_csv(path, parse_dates=["ds"])
    print(f"[로드] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


def build_model() -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def to_pods(rps: float) -> int:
    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))


def print_metrics(result: pd.DataFrame, label: str):
    smape     = (abs(result["y"] - result["yhat"]) / ((abs(result["y"]) + abs(result["yhat"])) / 2)).mean()
    pod_match = (result["y_pods"] == result["yhat_pods"]).mean()
    over      = (result["yhat_pods"] > result["y_pods"]).mean()
    under     = (result["yhat_pods"] < result["y_pods"]).mean()

    print(f"\n  [{label} 성능 요약]")
    print(f"  SMAPE:           {smape:.4f}  ({'✅ 통과' if smape < 0.15 else '❌ 미달'})")
    print(f"  Pod 정확도:      {pod_match:.1%}  (예측 = 실제)")
    print(f"  과잉 프로비저닝: {over:.1%}  (예측 > 실제, 비용 낭비)")
    print(f"  과소 프로비저닝: {under:.1%}  (예측 < 실제, 장애 위험)")


# ── 검증 1: 시각 검증 (전체 기간 fit) ─────────────────────────

def validate_fit(df: pd.DataFrame):
    """전체 2년치로 학습 후 동일 기간 재예측 → 실제 vs 예측 비교"""
    print("\n[시각 검증] 전체 기간 학습 후 재예측 중...")
    m = build_model()
    m.fit(df)

    forecast = m.predict(df[["ds", "is_monsoon", "typhoon_index"]])
    result = df[["ds", "y"]].copy()
    result["yhat"]       = forecast["yhat"].values
    result["yhat_lower"] = forecast["yhat_lower"].values
    result["yhat_upper"] = forecast["yhat_upper"].values
    result["y_pods"]     = result["y"].apply(to_pods)
    result["yhat_pods"]  = result["yhat"].apply(to_pods)

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max"}
    ).reset_index()

    monthly = result.groupby(result["ds"].dt.to_period("M")).agg(
        y_mean=("y", "mean"), yhat_mean=("yhat", "mean")
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle("Fit Check — Actual vs Predicted (2023–2024)", fontsize=14, fontweight="bold")

    # 상단: 일별 RPS
    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["y"],    color="steelblue", linewidth=1,   label="Actual RPS",    alpha=0.8)
    ax1.plot(daily["ds"], daily["yhat"], color="tomato",    linewidth=1,   label="Predicted RPS", linestyle="--")
    ax1.fill_between(daily["ds"], daily["yhat_lower"], daily["yhat_upper"], alpha=0.15, color="tomato")
    ax1.set_ylabel("RPS (req/min)")
    ax1.set_title("Daily Average RPS — Actual vs Predicted")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 중단: 월별 평균 막대
    ax2 = axes[1]
    x     = range(len(monthly))
    width = 0.4
    ax2.bar([i - width/2 for i in x], monthly["y_mean"],    width=width, label="Actual",    color="steelblue", alpha=0.8)
    ax2.bar([i + width/2 for i in x], monthly["yhat_mean"], width=width, label="Predicted", color="tomato",    alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(p) for p in monthly["ds"]], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Avg RPS (req/min)")
    ax2.set_title("Monthly Average — Actual vs Predicted")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # 하단: Pod 수
    ax3 = axes[2]
    ax3.step(daily_pods["ds"], daily_pods["y_pods"],    color="steelblue", linewidth=1.2, where="post", label="Actual Pods")
    ax3.step(daily_pods["ds"], daily_pods["yhat_pods"], color="tomato",    linewidth=1.2, where="post", label="Predicted Pods", linestyle="--")
    ax3.set_ylim(0, MAX_PODS + 1)
    ax3.set_ylabel("Pod Count")
    ax3.set_title("Required Pod Count — Actual vs Predicted")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(date_loc)
    ax3.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "plot_validation_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    print_metrics(result, "시각 검증")


# ── 검증 2: 홀드아웃 (2023 학습 → 2024 예측) ──────────────────

def validate_holdout(df: pd.DataFrame):
    """2023만으로 학습 → 2024 예측 → 2024 실제값과 비교"""
    train = df[df["ds"].dt.year == 2023].copy()
    test  = df[df["ds"].dt.year == 2024].copy()

    print(f"\n[홀드아웃 검증] 학습: {len(train):,}행 (2023) / 테스트: {len(test):,}행 (2024)")

    m = build_model()
    m.fit(train)

    forecast = m.predict(test[["ds", "is_monsoon", "typhoon_index"]])
    result = test[["ds", "y"]].copy()
    result["yhat"]       = forecast["yhat"].values
    result["yhat_lower"] = forecast["yhat_lower"].values
    result["yhat_upper"] = forecast["yhat_upper"].values
    result["y_pods"]     = result["y"].apply(to_pods)
    result["yhat_pods"]  = result["yhat"].apply(to_pods)
    result["pod_error"]  = (result["yhat_pods"] - result["y_pods"]).abs()

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max", "pod_error": "mean"}
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle("Holdout Validation — Train: 2023 / Test: 2024", fontsize=14, fontweight="bold")

    # 상단: RPS
    ax1 = axes[0]
    ax1.plot(daily["ds"], daily["y"],    color="steelblue", linewidth=1,   label="Actual RPS (2024)",    alpha=0.8)
    ax1.plot(daily["ds"], daily["yhat"], color="tomato",    linewidth=1,   label="Predicted RPS",        linestyle="--")
    ax1.fill_between(daily["ds"], daily["yhat_lower"], daily["yhat_upper"], alpha=0.15, color="tomato")
    ax1.set_ylabel("RPS (req/min)")
    ax1.set_title("Holdout: Daily Average RPS — Actual vs Predicted (2024)")
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 중단: Pod 수
    ax2 = axes[1]
    ax2.step(daily_pods["ds"], daily_pods["y_pods"],    color="steelblue", linewidth=1.5, where="post", label="Actual Pods")
    ax2.step(daily_pods["ds"], daily_pods["yhat_pods"], color="tomato",    linewidth=1.5, where="post", label="Predicted Pods", linestyle="--")
    ax2.set_ylim(0, MAX_PODS + 1)
    ax2.set_ylabel("Pod Count")
    ax2.set_title("Holdout: Required Pod Count — Actual vs Predicted (2024)")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(date_loc)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 하단: Pod 오차
    ax3 = axes[2]
    ax3.bar(daily_pods["ds"], daily_pods["pod_error"], color="orange", alpha=0.7, width=1, label="Pod Error (|actual - predicted|)")
    ax3.axhline(y=1, color="red", linestyle="--", linewidth=1, label="Error = 1 pod")
    ax3.set_ylabel("Pod Error")
    ax3.set_title("Holdout: Daily Pod Count Error  (0 = perfect)")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(date_loc)
    ax3.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = DATA_DIR / "plot_validation_holdout.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    print_metrics(result, "홀드아웃")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  모델 검증  (시각 검증 + 홀드아웃)")
    print("=" * 55)

    df = load_data()
    validate_fit(df)
    validate_holdout(df)

    print("\n생성된 파일:")
    print("  data/plot_validation_fit.png      — 전체 기간 실제 vs 예측")
    print("  data/plot_validation_holdout.png  — 홀드아웃 실제 vs 예측")


if __name__ == "__main__":
    main()
