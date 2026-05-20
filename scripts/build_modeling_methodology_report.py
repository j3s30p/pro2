from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

MODEL_RESULT_PATH = ROOT / "outputs" / "churn_model_final_result.csv"
THRESHOLD_RESULT_PATH = ROOT / "outputs" / "churn_threshold_result.csv"
RISK_SCORE_PATH = ROOT / "outputs" / "churn_risk_scores_test.csv"
REPORT_PATH = ROOT / "reports" / "modeling_methodology.html"
INDEX_PATH = ROOT / "reports" / "index.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dataframe_to_html(df: pd.DataFrame, classes: str = "data-table") -> str:
    return df.to_html(index=False, classes=classes, border=0, escape=False)


def make_model_chart(model_result: pd.DataFrame) -> str:
    sorted_df = model_result.sort_values("pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.barh(sorted_df["model"], sorted_df["pr_auc"], color="#2563eb", alpha=0.86, label="PR AUC")
    ax.scatter(sorted_df["recall"], sorted_df["model"], color="#dc2626", label="Recall", zorder=3)
    ax.set_title("Holdout Performance: PR AUC and Recall")
    ax.set_xlabel("Score")
    ax.set_xlim(0.70, 0.86)
    ax.legend(loc="lower right")
    for _, row in sorted_df.iterrows():
        ax.text(row["pr_auc"] + 0.002, row["model"], f"{row['pr_auc']:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_threshold_chart(threshold: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.plot(threshold["threshold"], threshold["precision"], label="Precision", color="#2563eb", linewidth=2)
    ax.plot(threshold["threshold"], threshold["recall"], label="Recall", color="#16a34a", linewidth=2)
    ax.plot(threshold["threshold"], threshold["f1"], label="F1", color="#dc2626", linewidth=2)
    best = threshold.sort_values("f1", ascending=False).iloc[0]
    ax.axvline(best["threshold"], color="#475569", linestyle="--", linewidth=1.4)
    ax.text(best["threshold"] + 0.015, 0.23, f"Selected threshold {best['threshold']:.2f}", fontsize=9)
    ax.set_title("Threshold Trade-off")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_feature_chart(feature_summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    labels = feature_summary["experiment"]
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], feature_summary["pr_auc"], width=width, color="#2563eb", label="PR AUC")
    ax.bar([i + width / 2 for i in x], feature_summary["top10_churn_rate"], width=width, color="#16a34a", label="Top 10% churn rate")
    ax.set_title("Feature Engineering Experiments")
    ax.set_ylabel("Score")
    ax.set_ylim(0.72, 0.90)
    ax.set_xticks(list(x), labels, rotation=12, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_index() -> None:
    html = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Netflix Churn Analysis Reports</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }
    main { max-width: 980px; margin: 0 auto; padding: 56px 24px; }
    h1 { font-size: 34px; margin: 0 0 12px; letter-spacing: 0; }
    p { color: #5b6472; line-height: 1.6; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 28px; }
    a.card { display: block; color: inherit; text-decoration: none; background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 22px; }
    a.card:hover { border-color: #2563eb; }
    .label { color: #2563eb; font-size: 13px; font-weight: 700; text-transform: uppercase; }
    h2 { margin: 8px 0; font-size: 22px; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>Netflix Churn Analysis Reports</h1>
    <p>
      모델링 흐름과 비즈니스 액션을 분리한 HTML 리포트입니다.
      첫 번째 페이지는 피처/모델/평가지표 선택 근거를, 두 번째 페이지는 상위 risk 고객의 집중 캠페인 전략을 설명합니다.
    </p>
    <div class="grid">
      <a class="card" href="modeling_methodology.html">
        <span class="label">Analysis Process</span>
        <h2>Modeling Methodology</h2>
        <p>피처 선택, 피처 엔지니어링 실험, 모델 후보 비교, PR AUC와 recall 중심 평가 이유.</p>
      </a>
      <a class="card" href="top_risk_retention_strategy.html">
        <span class="label">Business Insight</span>
        <h2>Retention Strategy</h2>
        <p>상위 10% risk 고객 중 성장기/장기 Basic 고객을 핵심 타겟으로 선정한 캠페인 전략.</p>
      </a>
    </div>
  </main>
</body>
</html>
"""
    INDEX_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_result = pd.read_csv(MODEL_RESULT_PATH)
    threshold = pd.read_csv(THRESHOLD_RESULT_PATH)
    risk_scores = pd.read_csv(RISK_SCORE_PATH)

    best_model = model_result.sort_values("pr_auc", ascending=False).iloc[0]
    best_threshold = threshold.sort_values("f1", ascending=False).iloc[0]
    base_churn_rate = risk_scores["actual_churn"].mean()
    top10 = risk_scores.sort_values("churn_probability", ascending=False).head(int(len(risk_scores) * 0.1))
    top10_churn_rate = top10["actual_churn"].mean()

    model_display = model_result.sort_values("pr_auc", ascending=False).assign(
        threshold=model_result["threshold"].map(lambda v: num(v, 2)),
        accuracy=model_result["accuracy"].map(pct),
        precision=model_result["precision"].map(pct),
        recall=model_result["recall"].map(pct),
        f1=model_result["f1"].map(num),
        roc_auc=model_result["roc_auc"].map(num),
        pr_auc=model_result["pr_auc"].map(num),
    ).rename(
        columns={
            "model": "Model",
            "threshold": "Threshold",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC AUC",
            "pr_auc": "PR AUC",
        }
    )

    threshold_display = pd.DataFrame(
        [
            {
                "Selection": "Default threshold",
                "Threshold": "0.50",
                "Precision": pct(threshold.loc[(threshold["threshold"] - 0.5).abs().idxmin(), "precision"]),
                "Recall": pct(threshold.loc[(threshold["threshold"] - 0.5).abs().idxmin(), "recall"]),
                "F1": num(threshold.loc[(threshold["threshold"] - 0.5).abs().idxmin(), "f1"]),
            },
            {
                "Selection": "F1-balanced threshold",
                "Threshold": num(best_threshold["threshold"], 2),
                "Precision": pct(best_threshold["precision"]),
                "Recall": pct(best_threshold["recall"]),
                "F1": num(best_threshold["f1"]),
            },
        ]
    )

    feature_summary = pd.DataFrame(
        [
            {
                "experiment": "Original Stacking",
                "feature_count": 18,
                "pr_auc": 0.7642,
                "selected_f1": 0.6938,
                "top10_churn_rate": 0.872,
                "decision": "Final risk score 기준으로 유지",
            },
            {
                "experiment": "FE First Try",
                "feature_count": 47,
                "pr_auc": 0.7638,
                "selected_f1": 0.6933,
                "top10_churn_rate": 0.869,
                "decision": "성능 개선 근거 부족",
            },
            {
                "experiment": "Behavior FE Pipeline",
                "feature_count": 53,
                "pr_auc": 0.7593,
                "selected_f1": 0.0,
                "top10_churn_rate": 0.858,
                "decision": "모델 입력보다 사후 해석용으로 활용",
            },
            {
                "experiment": "Paper-Based FE",
                "feature_count": 45,
                "pr_auc": 0.7545,
                "selected_f1": 0.6858,
                "top10_churn_rate": 0.0,
                "decision": "성능 개선은 없음, 논문 기반 해석 근거로 활용",
            },
            {
                "experiment": "Compact FE",
                "feature_count": 34,
                "pr_auc": 0.0,
                "selected_f1": 0.6856,
                "top10_churn_rate": 0.0,
                "decision": "CatBoost F1에는 도움, ranking 최종 근거는 약함",
            },
        ]
    )

    feature_display = feature_summary.copy()
    feature_display["pr_auc"] = feature_display["pr_auc"].map(lambda v: "-" if v == 0 else num(v))
    feature_display["selected_f1"] = feature_display["selected_f1"].map(lambda v: "-" if v == 0 else num(v))
    feature_display["top10_churn_rate"] = feature_display["top10_churn_rate"].map(lambda v: "-" if v == 0 else pct(v))
    feature_display = feature_display.rename(
        columns={
            "experiment": "Experiment",
            "feature_count": "Feature count",
            "pr_auc": "PR AUC",
            "selected_f1": "Best/selected F1",
            "top10_churn_rate": "Top 10% churn rate",
            "decision": "Decision",
        }
    )

    paper_feature_map = pd.DataFrame(
        [
            {
                "Paper factor": "가격 부담 / 가격 대비 가치",
                "Proxy features": "watch_time_per_plan, sessions_per_plan, value_for_money_score, price_burden_proxy",
                "Observed result": "가격 대비 가치 계열은 해석에는 유용했지만, 최종 성능은 original보다 낮음",
            },
            {
                "Paper factor": "추천 시스템 / 개인화 불만족",
                "Proxy features": "recommendation_effectiveness, recommendation_mismatch, personalization_risk",
                "Observed result": "추천 효과와 개인화 위험은 churn 방향 진단에 활용 가능",
            },
            {
                "Paper factor": "전환 위험 / 서비스 필요성 저하",
                "Proxy features": "switching_risk_proxy, service_dependency_score, low_service_need",
                "Observed result": "low_service_need와 switching_risk_proxy가 churn 방향과 잘 맞음",
            },
            {
                "Paper factor": "구독 피로감",
                "Proxy features": "subscription_fatigue_proxy, usage_per_account_age",
                "Observed result": "장기 가입 + 최근 저활동 패턴을 설명하는 보조 신호로 적합",
            },
            {
                "Paper factor": "몰아보기 / 콘텐츠 고갈",
                "Proxy features": "avg_minutes_per_watch_session, binge_intensity_proxy, content_exhaustion_proxy",
                "Observed result": "현재 데이터에서는 성능 개선보다는 행동 해석용 의미가 큼",
            },
        ]
    )

    paper_result = pd.DataFrame(
        [
            {
                "Feature set": "Original",
                "Best run": "LogisticRegression__original",
                "Test F1": 0.6873,
                "Precision": 0.6870,
                "Recall": 0.6876,
                "ROC AUC": 0.9038,
                "PR AUC": 0.7549,
            },
            {
                "Feature set": "Paper FE",
                "Best run": "LogisticRegression__paper_fe",
                "Test F1": 0.6858,
                "Precision": 0.6997,
                "Recall": 0.6724,
                "ROC AUC": 0.9036,
                "PR AUC": 0.7545,
            },
        ]
    )
    paper_result_display = paper_result.assign(
        **{
            "Test F1": paper_result["Test F1"].map(num),
            "Precision": paper_result["Precision"].map(pct),
            "Recall": paper_result["Recall"].map(pct),
            "ROC AUC": paper_result["ROC AUC"].map(num),
            "PR AUC": paper_result["PR AUC"].map(num),
        }
    )

    article_refs = pd.DataFrame(
        [
            {
                "Reference": "OTT 서비스의 이용자는 왜 구독을 해지하는가?",
                "Link": '<a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11396704">DBpia</a>',
                "Used for": "습관적 이용 인식, 서비스 불만족, 구독료 부담, 몰아보기/콘텐츠 고갈 가설",
            },
            {
                "Reference": "구독 피로감과 가격 민감도가 OTT 서비스의 구독 해지 의도에 미치는 영향",
                "Link": '<a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12538822">DBpia</a>',
                "Used for": "구독 피로감, 가격 민감도, UX 조절 효과 가설",
            },
            {
                "Reference": "OTT 플랫폼 유형에 따른 구독 해지 결정요인 분석",
                "Link": '<a href="https://www.earticle.net/Article/A478290">eArticle</a>',
                "Used for": "글로벌 SVOD의 비용 부담, 추천 알고리즘 불만족, 플랫폼 유형별 해지 요인 가설",
            },
        ]
    )

    model_chart = make_model_chart(model_result)
    threshold_chart = make_threshold_chart(threshold)
    feature_chart = make_feature_chart(feature_summary[feature_summary["pr_auc"] > 0])

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Modeling Methodology</title>
  <style>
    :root {{
      --text: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --soft: #f6f8fb;
      --blue: #1d4ed8;
      --green: #15803d;
      --red: #b91c1c;
      --slate: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; background: #ffffff; }}
    header {{ padding: 48px 40px 28px; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }}
    nav {{ max-width: 1100px; margin: 0 auto 24px; display: flex; flex-wrap: wrap; gap: 10px; }}
    nav a {{ color: var(--slate); text-decoration: none; border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; background: #ffffff; font-size: 14px; }}
    nav a.active {{ color: #ffffff; background: var(--blue); border-color: var(--blue); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 28px 64px; }}
    h1 {{ max-width: 1100px; margin: 0 auto 12px; font-size: 34px; line-height: 1.18; letter-spacing: 0; }}
    .subtitle {{ max-width: 1100px; margin: 0 auto; color: var(--muted); font-size: 16px; }}
    h2 {{ margin: 36px 0 14px; padding-top: 8px; font-size: 23px; letter-spacing: 0; }}
    h3 {{ margin: 24px 0 8px; font-size: 17px; }}
    p {{ margin: 8px 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0 28px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px 14px 12px; background: #ffffff; }}
    .metric .label {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric .value {{ display: block; margin-top: 5px; font-size: 24px; font-weight: 700; }}
    .note {{ border-left: 4px solid var(--blue); padding: 12px 14px; background: #eff6ff; color: #1e3a8a; margin: 16px 0 24px; }}
    .chart {{ margin: 14px 0 24px; border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #ffffff; }}
    .chart img {{ display: block; width: 100%; height: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0 22px; font-size: 14px; }}
    .data-table th {{ text-align: left; border-bottom: 2px solid var(--slate); padding: 9px 8px; white-space: nowrap; }}
    .data-table td {{ border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }}
    .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }}
    .box {{ border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: var(--soft); }}
    .box strong {{ display: block; margin-bottom: 8px; color: var(--slate); }}
    ul {{ margin: 8px 0 0; padding-left: 20px; }}
    li {{ margin: 4px 0; }}
    footer {{ margin-top: 42px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
    @media (max-width: 820px) {{
      header {{ padding: 36px 22px 24px; }}
      main {{ padding: 24px 18px 48px; }}
      h1 {{ font-size: 28px; }}
      .metrics, .two-col {{ grid-template-columns: 1fr; }}
      .data-table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <header>
    <nav>
      <a href="index.html">Report Home</a>
      <a class="active" href="modeling_methodology.html">Modeling Methodology</a>
      <a href="top_risk_retention_strategy.html">Retention Strategy</a>
    </nav>
    <h1>Churn Modeling & Feature Engineering Methodology</h1>
    <p class="subtitle">
      어떤 피처를 선택했고, 왜 피처 엔지니어링을 했으며, 최종적으로 왜 원본 피처 기반 risk score를 유지했는지 설명하는 방법론 리포트입니다.
    </p>
  </header>

  <main>
    <section>
      <h2>1. 문제 정의와 평가 기준</h2>
      <div class="metrics">
        <div class="metric"><span class="label">Test churn rate</span><span class="value">{pct(base_churn_rate)}</span></div>
        <div class="metric"><span class="label">Best PR AUC</span><span class="value">{num(best_model["pr_auc"])}</span></div>
        <div class="metric"><span class="label">Selected threshold</span><span class="value">{num(best_threshold["threshold"], 2)}</span></div>
        <div class="metric"><span class="label">Top 10% churn rate</span><span class="value">{pct(top10_churn_rate)}</span></div>
      </div>
      <p>
        이 프로젝트의 목표는 단순히 churn 여부를 0/1로 맞히는 것이 아니라, 이탈 가능성이 높은 고객을 우선순위화해
        retention campaign에 활용하는 것입니다. 따라서 accuracy보다 churn 고객을 얼마나 잘 상위 risk 구간에 모으는지가 중요합니다.
      </p>
      <div class="note">
        PR AUC는 양성 클래스인 churn 고객을 얼마나 잘 구분하는지 보여주고,
        recall은 실제 이탈 고객을 얼마나 놓치지 않는지 보여줍니다.
        캠페인 비용과 대상 품질을 함께 고려할 때는 F1과 top-k lift를 보조 지표로 사용했습니다.
      </div>
    </section>

    <section>
      <h2>2. 피처 선택 원칙</h2>
      <p>
        원본 데이터에는 인구통계, 요금제, 기기, 콘텐츠 반응, 사용 빈도, 최근 접속 정보가 포함되어 있습니다.
        EDA와 상관관계 분석에서 churn과 가장 강하게 연결된 신호는 최근 로그인 공백, 시청량, 세션 수, completion rate,
        recommendation click rate, app rating이었습니다.
      </p>
      <div class="two-col">
        <div class="box">
          <strong>최종 모델 입력에 유지한 피처</strong>
          <ul>
            <li>행동 강도: 시청시간, 세션 수, 주간 시청 세션</li>
            <li>콘텐츠 반응: completion, rating, recommendation click</li>
            <li>최근성: days since last login</li>
            <li>상품/환경: subscription type, payment, device, genre, source</li>
          </ul>
        </div>
        <div class="box">
          <strong>신중하게 다룬 피처</strong>
          <ul>
            <li>user_id는 식별자라 제거</li>
            <li>region/gender는 EDA상 차이가 작아 우선순위 낮음</li>
            <li>cluster label은 해석에는 유용하지만 예측 성능 개선은 제한적</li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>3. 피처 엔지니어링 실험과 최종 판단</h2>
      <p>
        피처 엔지니어링은 고위험 고객의 행동 패턴을 모델에 명시적으로 제공하기 위해 시도했습니다.
        ratio, engagement score, risk flag, categorical interaction, leakage-safe behavior feature, compact feature,
        논문 기반 paper feature를 실험했습니다.
      </p>
      {dataframe_to_html(feature_display)}
      <div class="chart"><img alt="Feature engineering experiment comparison" src="data:image/png;base64,{feature_chart}" /></div>
      <p>
        결론은 “파생변수가 무의미하다”가 아니라, 현재 합성 데이터에서는 원본 피처가 이미 강한 신호를 담고 있어
        많은 파생변수를 추가해도 ranking 품질이 개선되지 않았다는 것입니다.
        따라서 최종 risk score 산출에는 원본 피처 기반 모델을 유지하고, behavior feature는 사후 profiling과 action matrix에 활용하는 판단이 합리적입니다.
      </p>
    </section>

    <section>
      <h2>4. 논문 기반 Feature Engineering</h2>
      <p>
        `07-5`에서는 OTT 해지 관련 선행연구를 바탕으로 가격 대비 가치, 추천/개인화 반응, 전환 위험,
        구독 피로감, 몰아보기/콘텐츠 고갈 proxy를 만들었습니다.
        직접 관측되지 않는 개념은 현재 데이터의 사용량, 요금제, 추천 클릭률, 완료율, 최근 접속일로 대체 표현했습니다.
      </p>
      {dataframe_to_html(paper_feature_map)}
      <p>
        성능 비교 결과, 논문 기반 피처는 최종 모델 성능을 끌어올리지는 못했습니다.
        가장 좋은 `paper_fe` 후보는 precision은 높였지만 recall이 낮아졌고, F1/ROC AUC/PR AUC는 original보다 근소하게 낮았습니다.
      </p>
      {dataframe_to_html(paper_result_display)}
      <div class="note">
        `paper_fe`의 의미는 성능 개선보다 해석력입니다.
        RFM 위험, 서비스 필요성 저하, 전환 위험, 추천/완료 반응 저하 같은 개념을 수치화해
        “왜 이 고객이 위험한가”를 설명하는 보조 진단 피처로 활용할 수 있습니다.
      </div>
      <h3>참고한 논문</h3>
      {dataframe_to_html(article_refs)}
    </section>

    <section>
      <h2>5. 모델링 진행 방식</h2>
      <p>
        모델링은 baseline에서 시작해 튜닝과 앙상블로 확장했습니다.
        Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost를 비교했고,
        이후 GridSearchCV/RandomizedSearchCV, Soft Voting, Stacking을 실험했습니다.
      </p>
      {dataframe_to_html(model_display)}
      <div class="chart"><img alt="Model performance comparison" src="data:image/png;base64,{model_chart}" /></div>
      <p>
        Stacking이 PR AUC 기준 가장 높았지만, Logistic Regression 대비 개선 폭은 약 0.001 수준으로 작았습니다.
        이는 복잡한 모델이 압도적으로 우수하다기보다, 현재 데이터의 churn 패턴이 원본 행동 피처만으로도 상당히 잘 설명된다는 의미에 가깝습니다.
      </p>
    </section>

    <section>
      <h2>6. Threshold와 Ranking을 분리해서 본 이유</h2>
      <p>
        threshold 0.5는 기본값일 뿐이고, 캠페인 운영 목적에 맞는 최적 기준이 아닐 수 있습니다.
        threshold를 올리면 precision은 올라가지만 recall은 낮아지고, threshold를 낮추면 더 많은 이탈자를 잡지만 오탐이 많아집니다.
      </p>
      {dataframe_to_html(threshold_display)}
      <div class="chart"><img alt="Threshold precision recall f1 curve" src="data:image/png;base64,{threshold_chart}" /></div>
      <p>
        최종 운영 관점에서는 하나의 threshold로 모든 의사결정을 끝내기보다,
        예산에 따라 상위 10%, 20%, 30% 고객을 순차적으로 타겟팅하는 ranking 방식이 더 실무적입니다.
      </p>
    </section>

    <section>
      <h2>7. 최종 결론</h2>
      <div class="note">
        최종 모델링 결론은 “원본 피처 기반 risk score를 사용하고, 피처 엔지니어링 결과는 비즈니스 해석과 캠페인 세분화에 활용한다”입니다.
      </div>
      <p>
        모델 성능은 이미 합성 데이터에서 높은 수준에 도달했기 때문에, 추가 성능 개선보다 중요한 것은
        risk ranking 결과를 어떻게 캠페인 우선순위와 맞춤형 retention 전략으로 번역하느냐입니다.
        이 결론은 별도 비즈니스 리포트인 Retention Strategy 페이지에서 이어집니다.
      </p>
    </section>

    <footer>
      Source files: notebooks/04-07, outputs/churn_model_final_result.csv, outputs/churn_threshold_result.csv, outputs/churn_risk_scores_test.csv
    </footer>
  </main>
</body>
</html>
"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    build_index()
    print(REPORT_PATH)
    print(INDEX_PATH)


if __name__ == "__main__":
    main()
