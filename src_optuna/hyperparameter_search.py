"""
Optuna 기반 하이퍼파라미터 최적화 (TPE 베이지안 최적화 + 멀티프로세스 병렬)

src_tuned 대비 개선점:
  - 그리드 서치(20개 조합) → Optuna TPE (40 trial, 유망 구간 집중 탐색)
  - 이산 탐색 공간 → 연속 탐색 공간 (log-uniform, uniform)
  - 탐색 파라미터 추가: holidays_prior_scale (한국 공휴일 가중치)

병렬화 전략:
  - Optuna ask()/tell() API + ProcessPoolExecutor
  - ask(): Optuna가 다음 탐색 파라미터를 제안
  - tell(): 워커가 반환한 SMAPE를 Optuna에 보고 → TPE 갱신
  - constant_liar=True: 배치 내 pending trial을 현재 best로 가정 → 탐색 다양성 확보
  - study.optimize(n_jobs=...)는 threading 기반이라 Prophet에 비효율 → 이 방식 사용

탐색 파라미터:
  - changepoint_prior_scale  : log-uniform [0.001, 0.5]  — 트렌드 유연성
  - seasonality_prior_scale  : uniform     [1.0,  30.0]  — 계절성 강도
  - holidays_prior_scale     : uniform     [1.0,  20.0]  — 공휴일 가중치 (신규)

실행: python src_optuna/hyperparameter_search.py
출력: data/optuna_search_results.csv   — trial별 파라미터 및 SMAPE
      data/optuna_study.pkl            — Optuna study 객체
"""
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

N_TRIALS  = 40
N_WORKERS = max(1, os.cpu_count() - 1)
BATCH_SIZE = N_WORKERS


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dummy_request_rate.csv", parse_dates=["ds"])
    print(f"[로드] {len(df):,}행  ({df['ds'].min().date()} ~ {df['ds'].max().date()})")
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"])
    return df


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


# 모듈 최상위 함수여야 ProcessPoolExecutor에서 pickle 가능
def evaluate_params(args: tuple) -> float:
    """단일 파라미터 조합 평가 — 워커 프로세스에서 실행"""
    cps, sps, hps, df_log = args
    warnings.filterwarnings("ignore")

    m = build_model(cps, sps, hps)
    m.fit(df_log)

    df_cv   = cross_validation(m, initial="365 days", period="30 days",
                               horizon="7 days", disable_tqdm=True)
    df_perf = performance_metrics(df_cv, rolling_window=1)

    for col in ["smape", "mape", "mdape"]:
        if col in df_perf.columns:
            return float(df_perf[col].mean())

    return float("inf")


def search(df: pd.DataFrame):
    df_log = log_transform(df)

    print(f"\n[탐색] Optuna TPE  n_trials={N_TRIALS}  병렬 워커={N_WORKERS}개  배치={BATCH_SIZE}")
    print(f"  탐색 공간:")
    print(f"    changepoint_prior_scale : log-uniform [0.001, 0.500]")
    print(f"    seasonality_prior_scale : uniform     [1.0,  30.0]")
    print(f"    holidays_prior_scale    : uniform     [1.0,  20.0]")
    print()
    print(f"  {'Trial':>6}  {'cps':>7}  {'sps':>6}  {'hps':>6}  {'SMAPE':>8}")
    print(f"  {'-'*44}")

    # constant_liar=True: 배치 내 미완료 trial을 현재 best로 가정해 탐색 다양성 유지
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, constant_liar=True),
    )

    completed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        while completed < N_TRIALS:
            batch_size = min(BATCH_SIZE, N_TRIALS - completed)

            # Optuna에 배치만큼 trial 파라미터 요청 (결과 보고 전)
            trials, args = [], []
            for _ in range(batch_size):
                trial = study.ask()
                cps   = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
                sps   = trial.suggest_float("seasonality_prior_scale", 1.0, 30.0)
                hps   = trial.suggest_float("holidays_prior_scale",    1.0, 20.0)
                trials.append(trial)
                args.append((cps, sps, hps, df_log))

            # 배치 병렬 실행
            futures = {pool.submit(evaluate_params, a): trials[i] for i, a in enumerate(args)}

            for fut in as_completed(futures):
                trial = futures[fut]
                smape = fut.result()
                study.tell(trial, smape)   # 결과를 Optuna에 보고 → TPE 갱신
                completed += 1
                is_best = smape == study.best_value
                mark    = " ★" if is_best else ""
                p       = trial.params
                print(f"  [{completed:>2}/{N_TRIALS}]"
                      f"  {p['changepoint_prior_scale']:>7.4f}"
                      f"  {p['seasonality_prior_scale']:>6.2f}"
                      f"  {p['holidays_prior_scale']:>6.2f}"
                      f"  {smape:>8.4f}{mark}")

    rows = [
        {
            "trial":                    t.number,
            "changepoint_prior_scale":  t.params["changepoint_prior_scale"],
            "seasonality_prior_scale":  t.params["seasonality_prior_scale"],
            "holidays_prior_scale":     t.params["holidays_prior_scale"],
            "smape":                    t.value,
        }
        for t in study.trials
    ]
    results = pd.DataFrame(rows).sort_values("smape").reset_index(drop=True)

    out_csv = DATA_DIR / "optuna_search_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n[저장] {out_csv.name}")

    out_pkl = DATA_DIR / "optuna_study.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(study, f)
    print(f"[저장] {out_pkl.name}")

    return results, study


def main():
    print("=" * 60)
    print("  Optuna 하이퍼파라미터 탐색  (TPE 베이지안 최적화)")
    print("=" * 60)

    df             = load_data()
    results, study = search(df)
    best           = results.iloc[0]

    print("\n" + "=" * 60)
    print("  탐색 결과 (상위 5개)")
    print("=" * 60)
    print(results.head(5).to_string(index=False))
    print()
    print("  ★ 최적 파라미터")
    print(f"    changepoint_prior_scale = {best['changepoint_prior_scale']:.4f}")
    print(f"    seasonality_prior_scale = {best['seasonality_prior_scale']:.4f}")
    print(f"    holidays_prior_scale    = {best['holidays_prior_scale']:.4f}")
    print(f"    SMAPE                   = {best['smape']:.4f}")
    print()
    print("  → src_optuna/train_predict.py 에 이 값이 자동 반영됩니다.")


if __name__ == "__main__":
    main()
