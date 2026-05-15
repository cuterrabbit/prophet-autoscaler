"""
튜닝 버전: 모델 검증 (로그 변환 + 하이퍼파라미터 최적화)

비교 대상:
  - baseline   : src/validate.py 결과 (하드코딩)
  - log 변환   : src_optimized/validate.py 결과 (하드코딩)
  - tuned      : 현재 스크립트 실행 결과 (실시간 계산)

실행: python src_tuned/validate.py
출력: data/plot_tuned_validation_fit.png
      data/plot_tuned_validation_holdout.png
"""
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
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

# 이전 버전 실측 성능
VERSIONS = {
    "baseline": {
        "fit_smape":       0.6957, "fit_pod_acc":     0.682,
        "holdout_smape":   0.7566, "holdout_pod_acc": 0.698,
        "fit_over":        0.134,  "fit_under":       0.184,
        "holdout_over":    0.054,  "holdout_under":   0.172,
    },
    "log 변환": {
        "fit_smape":       0.5052, "fit_pod_acc":     0.746,
        "holdout_smape":   0.6030, "holdout_pod_acc": 0.712,
        "fit_over":        0.092,  "fit_under":       0.162,
        "holdout_over":    0.018,  "holdout_under":   0.270,
    },
}


# ── 공통 ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dummy_request_rate.csv", parse_dates=["ds"])
    print(f"[로드] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


def load_best_params() -> tuple[float, float]:
    path = DATA_DIR / "hp_search_results.csv"
    if path.exists():
        df  = pd.read_csv(path)
        row = df.sort_values("smape").iloc[0]
        cps = float(row["changepoint_prior_scale"])
        sps = float(row["seasonality_prior_scale"])
        print(f"[파라미터] cps={cps}  sps={sps}  (hp_search_results.csv)")
    else:
        cps, sps = 0.1, 10.0
        print(f"[파라미터] 탐색 결과 없음 — 기본값 사용  cps={cps}  sps={sps}")
    return cps, sps


def build_model(cps: float, sps: float) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def to_pods(rps: float) -> int:
    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"])
    return df


def inverse_transform(forecast: pd.DataFrame) -> pd.DataFrame:
    fc = forecast.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in fc.columns:
            fc[col] = np.expm1(fc[col])
    return fc


def calc_metrics(result: pd.DataFrame) -> dict:
    smape     = (abs(result["y"] - result["yhat"]) /
                 ((abs(result["y"]) + abs(result["yhat"])) / 2)).mean()
    pod_match = (result["y_pods"] == result["yhat_pods"]).mean()
    over      = (result["yhat_pods"] > result["y_pods"]).mean()
    under     = (result["yhat_pods"] < result["y_pods"]).mean()
    return {"smape": smape, "pod_acc": pod_match, "over": over, "under": under}


def print_3way(tuned: dict, label: str):
    b = VERSIONS["baseline"]
    l = VERSIONS["log 변환"]
    smape_k   = f"{label.lower()}_smape"
    pod_acc_k = f"{label.lower()}_pod_acc"
    over_k    = f"{label.lower()}_over"
    under_k   = f"{label.lower()}_under"

    print(f"\n  [{label}]")
    print(f"  {'항목':<22} {'baseline':>10} {'log 변환':>10} {'tuned':>10}")
    print(f"  {'-'*54}")

    def row(name, b_val, l_val, t_val, fmt=".4f"):
        arrow = "↑" if t_val > l_val else ("↓" if t_val < l_val else "=")
        print(f"  {name:<22} {b_val:>10{fmt}} {l_val:>10{fmt}} {t_val:>10{fmt}} {arrow}")

    row("SMAPE",             b[smape_k],   l[smape_k],   tuned["smape"])
    row("Pod 정확도",        b[pod_acc_k], l[pod_acc_k], tuned["pod_acc"], ".3f")
    row("과잉 프로비저닝",   b[over_k],    l[over_k],    tuned["over"],   ".3f")
    row("과소 프로비저닝",   b[under_k],   l[under_k],   tuned["under"],  ".3f")


# ── 검증 1: 시각 검증 ─────────────────────────────────────────

def validate_fit(df: pd.DataFrame, cps: float, sps: float) -> dict:
    print("\n[시각 검증] 전체 기간 학습 후 재예측 중...")
    m = build_model(cps, sps)
    m.fit(log_transform(df))

    raw      = m.predict(df[["ds", "is_monsoon", "typhoon_index"]])
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    result                = df[["ds", "y"]].copy()
    result["yhat"]        = forecast["yhat"].values
    result["yhat_lower"]  = forecast["yhat_lower"].values
    result["yhat_upper"]  = forecast["yhat_upper"].values
    result["y_pods"]      = result["y"].apply(to_pods)
    result["yhat_pods"]   = result["yhat"].apply(to_pods)

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max"}
    ).reset_index()
    monthly    = result.groupby(result["ds"].dt.to_period("M")).agg(
        y_mean=("y", "mean"), yhat_mean=("yhat", "mean")
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle(f"Tuned Fit Check (cps={cps}, sps={sps}) — Actual vs Predicted 2023–2024",
                 fontsize=13, fontweight="bold")

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

    ax2 = axes[1]
    x, width = range(len(monthly)), 0.4
    ax2.bar([i - width/2 for i in x], monthly["y_mean"],    width=width, label="Actual",    color="steelblue", alpha=0.8)
    ax2.bar([i + width/2 for i in x], monthly["yhat_mean"], width=width, label="Predicted", color="tomato",    alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(p) for p in monthly["ds"]], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Avg RPS (req/min)")
    ax2.set_title("Monthly Average — Actual vs Predicted")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

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
    out = DATA_DIR / "plot_tuned_validation_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    metrics = calc_metrics(result)
    print_3way(metrics, "fit")
    return metrics


# ── 검증 2: 홀드아웃 ──────────────────────────────────────────

def validate_holdout(df: pd.DataFrame, cps: float, sps: float) -> dict:
    train = df[df["ds"].dt.year == 2023].copy()
    test  = df[df["ds"].dt.year == 2024].copy()

    print(f"\n[홀드아웃 검증] 학습: {len(train):,}행 (2023) / 테스트: {len(test):,}행 (2024)")
    m = build_model(cps, sps)
    m.fit(log_transform(train))

    raw      = m.predict(test[["ds", "is_monsoon", "typhoon_index"]])
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    result                = test[["ds", "y"]].copy()
    result["yhat"]        = forecast["yhat"].values
    result["yhat_lower"]  = forecast["yhat_lower"].values
    result["yhat_upper"]  = forecast["yhat_upper"].values
    result["y_pods"]      = result["y"].apply(to_pods)
    result["yhat_pods"]   = result["yhat"].apply(to_pods)
    result["pod_error"]   = (result["yhat_pods"] - result["y_pods"]).abs()

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max", "pod_error": "mean"}
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle(f"Tuned Holdout (cps={cps}, sps={sps}) — Train: 2023 / Test: 2024",
                 fontsize=13, fontweight="bold")

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

    ax3 = axes[2]
    ax3.bar(daily_pods["ds"], daily_pods["pod_error"],
            color="orange", alpha=0.7, width=1, label="Pod Error (|actual - predicted|)")
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
    out = DATA_DIR / "plot_tuned_validation_holdout.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    metrics = calc_metrics(result)
    print_3way(metrics, "holdout")
    return metrics


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  튜닝 모델 검증  (baseline / log / tuned 3-way 비교)")
    print("=" * 55)

    cps, sps = load_best_params()
    df       = load_data()

    fit_m      = validate_fit(df, cps, sps)
    holdout_m  = validate_holdout(df, cps, sps)

    print("\n" + "=" * 55)
    print("  최종 요약")
    print("=" * 55)
    b = VERSIONS["baseline"]
    l = VERSIONS["log 변환"]
    print(f"  {'':22} {'baseline':>10} {'log 변환':>10} {'tuned':>10}")
    print(f"  {'-'*54}")
    print(f"  {'시각검증 Pod 정확도':<22} {b['fit_pod_acc']:>10.1%} {l['fit_pod_acc']:>10.1%} {fit_m['pod_acc']:>10.1%}")
    print(f"  {'홀드아웃 Pod 정확도':<22} {b['holdout_pod_acc']:>10.1%} {l['holdout_pod_acc']:>10.1%} {holdout_m['pod_acc']:>10.1%}")
    print(f"  {'홀드아웃 과소 프로비전':<22} {b['holdout_under']:>10.1%} {l['holdout_under']:>10.1%} {holdout_m['under']:>10.1%}")
    print()
    print("  생성된 파일:")
    print("    data/plot_tuned_validation_fit.png")
    print("    data/plot_tuned_validation_holdout.png")


if __name__ == "__main__":
    main()
