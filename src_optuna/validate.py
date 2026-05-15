"""
Optuna 버전: 모델 검증 (4-way 비교)

비교 대상:
  - baseline   : src/validate.py 결과 (하드코딩)
  - log 변환   : src_optimized/validate.py 결과 (하드코딩)
  - tuned      : src_tuned/validate.py 결과 (하드코딩)
  - optuna     : 현재 스크립트 실행 결과 (실시간 계산, yhat_upper Pod)

실행: python src_optuna/validate.py
출력: data/plot_optuna_validation_fit.png
      data/plot_optuna_validation_holdout.png
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
    "tuned": {
        "fit_smape":       0.5052, "fit_pod_acc":     0.746,
        "holdout_smape":   0.5974, "holdout_pod_acc": 0.714,
        "fit_over":        0.092,  "fit_under":       0.162,
        "holdout_over":    0.018,  "holdout_under":   0.268,
    },
}


# ── 공통 ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dummy_request_rate.csv", parse_dates=["ds"])
    print(f"[로드] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


def load_best_params() -> tuple[float, float, float]:
    path = DATA_DIR / "optuna_search_results.csv"
    if path.exists():
        df  = pd.read_csv(path)
        row = df.sort_values("smape").iloc[0]
        cps = float(row["changepoint_prior_scale"])
        sps = float(row["seasonality_prior_scale"])
        hps = float(row["holidays_prior_scale"])
        print(f"[파라미터] cps={cps:.4f}  sps={sps:.4f}  hps={hps:.4f}  (optuna_search_results.csv)")
    else:
        cps, sps, hps = 0.1, 10.0, 10.0
        print(f"[파라미터] 탐색 결과 없음 — 기본값 사용")
    return cps, sps, hps


def build_model(cps: float, sps: float, hps: float) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        holidays_prior_scale=hps,
    )
    m.add_country_holidays(country_name="KR")
    m.add_regressor("is_monsoon")
    m.add_regressor("typhoon_index")
    return m


def to_pods_yhat(rps: float) -> int:
    """yhat 기준 Pod (비교용)"""
    eff = CAPACITY_PER_POD * (1 - SAFETY_MARGIN)
    return max(MIN_PODS, min(math.ceil(max(rps, 0) / eff), MAX_PODS))


def to_pods_upper(rps: float) -> int:
    """yhat_upper 기준 Pod — 과소 프로비저닝 억제"""
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


def calc_metrics(result: pd.DataFrame, pod_col: str = "yhat_pods") -> dict:
    smape     = (abs(result["y"] - result["yhat"]) /
                 ((abs(result["y"]) + abs(result["yhat"])) / 2)).mean()
    pod_match = (result["y_pods"] == result[pod_col]).mean()
    over      = (result[pod_col] > result["y_pods"]).mean()
    under     = (result[pod_col] < result["y_pods"]).mean()
    return {"smape": smape, "pod_acc": pod_match, "over": over, "under": under}


def print_4way(optuna_m: dict, label: str):
    b  = VERSIONS["baseline"]
    l  = VERSIONS["log 변환"]
    t  = VERSIONS["tuned"]
    sk = label.lower()

    print(f"\n  [{label}]")
    print(f"  {'항목':<22} {'baseline':>10} {'log 변환':>10} {'tuned':>10} {'optuna':>10}")
    print(f"  {'-'*66}")

    def row(name, b_v, l_v, t_v, o_v, fmt=".4f"):
        arrow = "↑" if o_v > t_v else ("↓" if o_v < t_v else "=")
        print(f"  {name:<22} {b_v:>10{fmt}} {l_v:>10{fmt}} {t_v:>10{fmt}} {o_v:>10{fmt}} {arrow}")

    row("SMAPE",             b[f"{sk}_smape"],   l[f"{sk}_smape"],   t[f"{sk}_smape"],   optuna_m["smape"])
    row("Pod 정확도",        b[f"{sk}_pod_acc"], l[f"{sk}_pod_acc"], t[f"{sk}_pod_acc"], optuna_m["pod_acc"], ".3f")
    row("과잉 프로비저닝",   b[f"{sk}_over"],    l[f"{sk}_over"],    t[f"{sk}_over"],    optuna_m["over"],   ".3f")
    row("과소 프로비저닝",   b[f"{sk}_under"],   l[f"{sk}_under"],   t[f"{sk}_under"],   optuna_m["under"],  ".3f")


# ── 검증 1: 시각 검증 ─────────────────────────────────────────

def validate_fit(df: pd.DataFrame, cps: float, sps: float, hps: float) -> dict:
    print("\n[시각 검증] 전체 기간 학습 후 재예측 중...")
    m = build_model(cps, sps, hps)
    m.fit(log_transform(df))

    raw      = m.predict(df[["ds", "is_monsoon", "typhoon_index"]])
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    result                 = df[["ds", "y"]].copy()
    result["yhat"]         = forecast["yhat"].values
    result["yhat_lower"]   = forecast["yhat_lower"].values
    result["yhat_upper"]   = forecast["yhat_upper"].values
    result["y_pods"]       = result["y"].apply(to_pods_yhat)
    result["yhat_pods"]    = result["yhat"].apply(to_pods_yhat)
    result["yhat_u_pods"]  = result["yhat_upper"].apply(to_pods_upper)

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max", "yhat_u_pods": "max"}
    ).reset_index()
    monthly    = result.groupby(result["ds"].dt.to_period("M")).agg(
        y_mean=("y", "mean"), yhat_mean=("yhat", "mean")
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle(
        f"Optuna Fit Check (cps={cps:.4f}, sps={sps:.4f}, hps={hps:.4f}) — yhat_upper Pod",
        fontsize=13, fontweight="bold"
    )

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
    ax3.step(daily_pods["ds"], daily_pods["y_pods"],      color="steelblue", linewidth=1.2, where="post", label="Actual Pods")
    ax3.step(daily_pods["ds"], daily_pods["yhat_pods"],   color="gray",      linewidth=1.0, where="post", label="Pods (yhat)",       linestyle="--", alpha=0.6)
    ax3.step(daily_pods["ds"], daily_pods["yhat_u_pods"], color="tomato",    linewidth=1.2, where="post", label="Pods (yhat_upper)", linestyle="-")
    ax3.set_ylim(0, MAX_PODS + 1)
    ax3.set_ylabel("Pod Count")
    ax3.set_title("Required Pod Count — Actual / yhat / yhat_upper 비교")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(date_loc)
    ax3.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = DATA_DIR / "plot_optuna_validation_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    metrics = calc_metrics(result, pod_col="yhat_u_pods")
    print_4way(metrics, "fit")
    return metrics


# ── 검증 2: 홀드아웃 ──────────────────────────────────────────

def validate_holdout(df: pd.DataFrame, cps: float, sps: float, hps: float) -> dict:
    train = df[df["ds"].dt.year == 2023].copy()
    test  = df[df["ds"].dt.year == 2024].copy()

    print(f"\n[홀드아웃 검증] 학습: {len(train):,}행 (2023) / 테스트: {len(test):,}행 (2024)")
    m = build_model(cps, sps, hps)
    m.fit(log_transform(train))

    raw      = m.predict(test[["ds", "is_monsoon", "typhoon_index"]])
    forecast = inverse_transform(raw[["ds", "yhat", "yhat_lower", "yhat_upper"]])

    result                 = test[["ds", "y"]].copy()
    result["yhat"]         = forecast["yhat"].values
    result["yhat_lower"]   = forecast["yhat_lower"].values
    result["yhat_upper"]   = forecast["yhat_upper"].values
    result["y_pods"]       = result["y"].apply(to_pods_yhat)
    result["yhat_pods"]    = result["yhat"].apply(to_pods_yhat)
    result["yhat_u_pods"]  = result["yhat_upper"].apply(to_pods_upper)
    result["pod_error"]    = (result["yhat_u_pods"] - result["y_pods"]).abs()

    daily      = result.set_index("ds").resample("D").mean(numeric_only=True).reset_index()
    daily_pods = result.set_index("ds").resample("D").agg(
        {"y_pods": "max", "yhat_pods": "max", "yhat_u_pods": "max", "pod_error": "mean"}
    ).reset_index()

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle(
        f"Optuna Holdout (cps={cps:.4f}, sps={sps:.4f}, hps={hps:.4f}) — Train: 2023 / Test: 2024",
        fontsize=13, fontweight="bold"
    )

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
    ax2.step(daily_pods["ds"], daily_pods["y_pods"],      color="steelblue", linewidth=1.5, where="post", label="Actual Pods")
    ax2.step(daily_pods["ds"], daily_pods["yhat_pods"],   color="gray",      linewidth=1.0, where="post", label="Pods (yhat)",       linestyle="--", alpha=0.6)
    ax2.step(daily_pods["ds"], daily_pods["yhat_u_pods"], color="tomato",    linewidth=1.5, where="post", label="Pods (yhat_upper)", linestyle="-")
    ax2.set_ylim(0, MAX_PODS + 1)
    ax2.set_ylabel("Pod Count")
    ax2.set_title("Holdout: Pod Count — Actual / yhat / yhat_upper (2024)")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(date_loc)
    ax2.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.bar(daily_pods["ds"], daily_pods["pod_error"],
            color="orange", alpha=0.7, width=1, label="Pod Error (yhat_upper 기준)")
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
    out = DATA_DIR / "plot_optuna_validation_holdout.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {out.name}")

    metrics = calc_metrics(result, pod_col="yhat_u_pods")
    print_4way(metrics, "holdout")
    return metrics


# ── 메인 ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Optuna 모델 검증  (baseline / log / tuned / optuna 4-way)")
    print("=" * 60)

    cps, sps, hps = load_best_params()
    df            = load_data()

    fit_m     = validate_fit(df, cps, sps, hps)
    holdout_m = validate_holdout(df, cps, sps, hps)

    b = VERSIONS["baseline"]
    l = VERSIONS["log 변환"]
    t = VERSIONS["tuned"]

    print("\n" + "=" * 60)
    print("  최종 요약  (yhat_upper 기반 Pod)")
    print("=" * 60)
    print(f"  {'':26} {'baseline':>9} {'log':>9} {'tuned':>9} {'optuna':>9}")
    print(f"  {'-'*64}")
    print(f"  {'시각검증 Pod 정확도':<26} {b['fit_pod_acc']:>9.1%} {l['fit_pod_acc']:>9.1%} {t['fit_pod_acc']:>9.1%} {fit_m['pod_acc']:>9.1%}")
    print(f"  {'홀드아웃 Pod 정확도':<26} {b['holdout_pod_acc']:>9.1%} {l['holdout_pod_acc']:>9.1%} {t['holdout_pod_acc']:>9.1%} {holdout_m['pod_acc']:>9.1%}")
    print(f"  {'홀드아웃 과소 프로비전':<26} {b['holdout_under']:>9.1%} {l['holdout_under']:>9.1%} {t['holdout_under']:>9.1%} {holdout_m['under']:>9.1%}")
    print(f"  {'홀드아웃 과잉 프로비전':<26} {b['holdout_over']:>9.1%} {l['holdout_over']:>9.1%} {t['holdout_over']:>9.1%} {holdout_m['over']:>9.1%}")
    print()
    print("  생성된 파일:")
    print("    data/plot_optuna_validation_fit.png")
    print("    data/plot_optuna_validation_holdout.png")


if __name__ == "__main__":
    main()
