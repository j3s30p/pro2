from __future__ import annotations

import base64
import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

RISK_SCORE_PATH = ROOT / "outputs" / "churn_risk_scores_test.csv"
MODEL_RESULT_PATH = ROOT / "outputs" / "churn_model_final_result.csv"
THRESHOLD_RESULT_PATH = ROOT / "outputs" / "churn_threshold_result.csv"
ANALYSIS_PATH = ROOT / "reports" / "retention_analysis.html"
STRATEGY_PATH = ROOT / "reports" / "top_risk_retention_strategy.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def model_label(value: str) -> str:
    if value in {"Stacking", "Stacking_original_features"}:
        return "Stacking (original features)"
    return value.replace("_", " ")


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dataframe_to_html(df: pd.DataFrame, highlight_terms: tuple[str, ...] = ()) -> str:
    html = df.to_html(index=False, classes="data-table", border=0, escape=False)
    if not highlight_terms:
        return html

    def mark_row(match: re.Match[str]) -> str:
        row_html = match.group(0)
        should_highlight = False
        for term in highlight_terms:
            if "|" in term:
                should_highlight = all(part in row_html for part in term.split("|"))
            else:
                should_highlight = term in row_html
            if should_highlight:
                break
        if should_highlight:
            return row_html.replace("<tr>", '<tr class="highlight-row">', 1)
        return row_html

    return re.sub(r"<tr>\s*(?:<td>.*?</td>\s*)+</tr>", mark_row, html, flags=re.S)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top_pct"] = (df.index + 1) / len(df)
    df["risk_group"] = pd.cut(
        df["top_pct"],
        bins=[0, 0.10, 0.30, 1],
        labels=["High: top 10%", "Medium: 10-30%", "Low: 30-100%"],
        include_lowest=True,
    )
    df["tenure_segment"] = pd.cut(
        df["account_age_months"],
        bins=[0, 12, 36, 72, 10**9],
        labels=["New <=12m", "Growing 13-36m", "Long 37-72m", "Very long 73m+"],
        include_lowest=True,
    )
    df["activity_segment"] = pd.cut(
        df["days_since_last_login"],
        bins=[-1, 7, 30, 10**9],
        labels=["Active <=7d", "Watch 8-30d", "Dormant 31d+"],
    )
    return df


def build_topk(df: pd.DataFrame) -> pd.DataFrame:
    base_rate = df["actual_churn"].mean()
    rows = []
    for target_pct in [0.05, 0.10, 0.20, 0.30]:
        top = df.head(int(len(df) * target_pct))
        rows.append(
            {
                "target_pct": target_pct,
                "customers": len(top),
                "churn_rate": top["actual_churn"].mean(),
                "lift": top["actual_churn"].mean() / base_rate,
                "captured_churner": top["actual_churn"].sum() / df["actual_churn"].sum(),
            }
        )
    return pd.DataFrame(rows)


def make_topk_chart(topk: pd.DataFrame) -> str:
    labels = [f"Top {int(v * 100)}%" for v in topk["target_pct"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    axes[0].bar(labels, topk["churn_rate"] * 100, color="#1f5fbf")
    axes[0].set_title("Actual Churn Rate")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Churn rate (%)")
    for i, value in enumerate(topk["churn_rate"] * 100):
        axes[0].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    axes[1].bar(labels, topk["captured_churner"] * 100, color="#287a47")
    axes[1].set_title("Captured Churners")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Captured churners (%)")
    for i, value in enumerate(topk["captured_churner"] * 100):
        axes[1].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_tenure_chart(high_tenure: pd.DataFrame) -> str:
    labels = high_tenure["tenure_segment"].astype(str)
    fig, ax1 = plt.subplots(figsize=(10.2, 4.3))
    ax2 = ax1.twinx()
    ax1.bar(labels, high_tenure["customers"], color="#46566f", alpha=0.86)
    ax2.plot(labels, high_tenure["churn_rate"] * 100, color="#b42318", marker="o", linewidth=2.4)
    ax1.set_title("Top 10% Risk Customers by Tenure")
    ax1.set_ylabel("Customers")
    ax2.set_ylabel("Churn rate (%)")
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=12)
    for i, value in enumerate(high_tenure["customers"]):
        ax1.text(i, value + 8, f"{int(value)}", ha="center", fontsize=9)
    for i, value in enumerate(high_tenure["churn_rate"] * 100):
        ax2.text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9, color="#8a1c13")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_focus_profile_chart(feature_profile: pd.DataFrame) -> str:
    metrics = [
        ("days_since_last_login", "Days since login"),
        ("avg_watch_time_minutes_per_week", "Weekly watch min"),
        ("completion_rate", "Completion rate"),
        ("recommendation_click_rate", "Recommendation click rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.1))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        bars = ax.bar(feature_profile["group"], feature_profile[metric], color=["#7b8794", "#1f5fbf", "#b42318"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=10)
        ymax = max(feature_profile[metric]) or 1
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + ymax * 0.025, f"{height:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_cluster_risk_heatmap(cluster_risk: pd.DataFrame) -> str:
    pivot = cluster_risk.pivot(index="cluster", columns="risk_group", values="churn_rate")
    pivot = pivot[[col for col in ["High: top 10%", "Medium: 10-30%", "Low: 30-100%"] if col in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.imshow(pivot.values * 100, cmap="Reds", vmin=0, vmax=100)
    ax.set_title("Churn Rate: KMeans Cluster x Risk Group")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=12, ha="right")
    ax.set_yticks(range(len(pivot.index)), [f"Cluster {idx}" for idx in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j] * 100:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax).set_label("Churn rate (%)")
    fig.tight_layout()
    return fig_to_base64(fig)


def css() -> str:
    return """
    :root {
      --text: #121826;
      --muted: #5e6b7a;
      --line: #dbe2ea;
      --panel: #ffffff;
      --soft: #f6f8fb;
      --blue: #1f5fbf;
      --green: #287a47;
      --red: #b42318;
      --amber: #a65f00;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #ffffff; line-height: 1.56; }
    header { background: #f7f9fc; border-bottom: 1px solid var(--line); padding: 34px 32px 30px; }
    nav { max-width: 1120px; margin: 0 auto 22px; display: flex; flex-wrap: wrap; gap: 8px; }
    nav a { color: #334155; text-decoration: none; border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; background: #ffffff; font-size: 14px; }
    nav a.active { color: #ffffff; background: var(--blue); border-color: var(--blue); }
    .hero { max-width: 1120px; margin: 0 auto; display: grid; grid-template-columns: 1.4fr 0.9fr; gap: 20px; align-items: end; }
    h1 { margin: 0; font-size: 36px; line-height: 1.18; letter-spacing: 0; }
    .subtitle { margin: 12px 0 0; color: var(--muted); font-size: 16px; max-width: 760px; }
    .kpi-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .kpi, .card, .action, .evidence { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .kpi { padding: 14px; }
    .kpi span, .eyebrow { display: block; color: var(--muted); font-size: 13px; font-weight: 650; }
    .kpi b { display: block; margin-top: 4px; font-size: 24px; }
    main { max-width: 1120px; margin: 0 auto; padding: 30px 28px 64px; }
    section { margin: 0 0 42px; }
    h2 { margin: 0 0 14px; font-size: 24px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 18px; }
    p { margin: 8px 0 12px; }
    .section-lead { max-width: 820px; color: var(--muted); }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .card, .action, .evidence { padding: 18px; }
    .card strong, .action strong, .evidence strong { display: block; font-size: 19px; line-height: 1.28; margin: 4px 0 8px; }
    .card p, .action p, .evidence p { color: var(--muted); font-size: 14px; }
    .insight { border: 1px solid var(--line); border-radius: 8px; background: #ffffff; padding: 16px; }
    .insight b { display: block; margin: 4px 0 6px; font-size: 22px; line-height: 1.2; }
    .insight p { color: var(--muted); font-size: 14px; margin-bottom: 0; }
    .reading-guide { border-left: 4px solid var(--blue); background: #f0f6ff; border-radius: 8px; padding: 14px 16px; margin: 14px 0 16px; }
    .reading-guide strong { display: block; color: #174ea6; margin-bottom: 4px; }
    .caption { color: var(--muted); font-size: 13px; margin: -8px 0 12px; }
    .decision { border-left: 5px solid var(--red); background: #fff7f6; padding: 16px 18px; border-radius: 8px; margin: 16px 0 20px; }
    .decision strong { display: block; color: var(--red); margin-bottom: 6px; font-size: 18px; }
    .chart { border: 1px solid var(--line); border-radius: 8px; background: #ffffff; padding: 12px; margin: 14px 0 20px; }
    .chart img { display: block; width: 100%; height: auto; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; margin: 12px 0 22px; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 14px; background: #ffffff; }
    .data-table th { text-align: left; padding: 10px 9px; border-bottom: 2px solid #334155; white-space: nowrap; background: #fbfcfe; }
    .data-table td { padding: 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
    .data-table tr.highlight-row td { background: #fff7ed; font-weight: 650; }
    .data-table tbody tr:hover td { background: #f8fafc; }
    .data-table tr:last-child td { border-bottom: 0; }
    .small { color: var(--muted); font-size: 13px; }
    ul { margin: 8px 0 0; padding-left: 20px; }
    li { margin: 4px 0; }
    footer { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 18px; color: var(--muted); font-size: 13px; }
    @media (max-width: 860px) {
      header { padding: 28px 20px; }
      main { padding: 24px 18px 48px; }
      .hero, .grid-2, .grid-3, .kpi-row { grid-template-columns: 1fr; }
      h1 { font-size: 29px; }
    }
    """


def render_analysis(context: dict[str, object]) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Retention Analysis Evidence</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <nav>
      <a href="index.html">Report Home</a>
      <a href="eda_analysis.html">EDA Evidence</a>
      <a href="interactive_report.html">Interactive Report</a>
      <a href="modeling_methodology.html">Modeling Methodology</a>
      <a class="active" href="retention_analysis.html">Retention Analysis</a>
      <a href="top_risk_retention_strategy.html">Final Strategy</a>
    </nav>
    <div class="hero">
      <div>
        <h1>Retention Target Analysis</h1>
        <p class="subtitle">06 노트북의 가정, 실험, 결과, 최종 판단 구조를 유지해 risk score 기반 타겟팅의 타당성을 검증한 분석 페이지입니다.</p>
      </div>
      <div class="kpi-row">
        <div class="kpi"><span>Top 30% captured</span><b>{context["top30_captured"]}</b></div>
        <div class="kpi"><span>Top 10% churn</span><b>{context["top10_churn"]}</b></div>
        <div class="kpi"><span>Focus target</span><b>{context["focus_customers"]}</b></div>
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>1. 가정 - risk score ranking으로 캠페인 우선순위를 정할 수 있다</h2>
      <p class="section-lead">Top 30%는 모델 ranking의 설명력을 검증하는 범위이며, Top 10%는 비용이 수반되는 retention action의 우선 실행 범위로 해석합니다.</p>
      <div class="grid-3">
        <div class="insight"><span class="eyebrow">Ranking coverage</span><b>{context["top30_captured"]}</b><p>Top 30% 구간에 포함된 전체 이탈자 비중.</p></div>
        <div class="insight"><span class="eyebrow">Priority segment</span><b>{context["top10_churn"]}</b><p>Top 10% 구간의 실제 이탈률.</p></div>
        <div class="insight"><span class="eyebrow">Operating scope</span><b>Top 10%</b><p>초기 유료 캠페인 집행 범위.</p></div>
      </div>
      <div class="reading-guide">
        <strong>Percentile construction</strong>
        최종 모델 <code>{context["final_model"]}</code>이 산출한 <code>churn_probability</code>를 기준으로 test 고객 {context["total_customers"]}명을 내림차순 정렬한 뒤,
        상위 {context["top10_customers"]}, {context["top20_customers"]}, {context["top30_customers"]}명을 각각 Top 10%, Top 20%, Top 30%로 정의했습니다.
        threshold는 0/1 분류 성능 비교용이며, top-k ranking 구간 생성에는 사용하지 않았습니다.
      </div>
      <div class="reading-guide">
        <strong>Metric definition</strong>
        <code>Actual churn rate</code>는 해당 구간 내 실제 이탈자 수를 구간 고객 수로 나눈 값입니다.
        <code>Captured churners</code>는 해당 구간 내 실제 이탈자 수를 전체 test 이탈자 {context["total_churners"]}명으로 나눈 값이며,
        <code>Lift</code>는 전체 test churn rate 대비 해당 구간 churn rate의 배율입니다.
      </div>
      <div class="table-wrap">{context["topk_table"]}</div>
      <p class="caption">Top 30%는 coverage, Top 10%는 execution quality 관점에서 해석합니다.</p>
      <div class="chart"><img alt="Top-k targeting performance" src="data:image/png;base64,{context["topk_chart"]}" /></div>
    </section>
    <section>
      <h2>2. 실험 - Top 10% 내부 세분화</h2>
      <p class="section-lead">상위 10% 안에서도 성장기와 장기 고객의 규모가 크고 이탈률도 높습니다. 이 두 구간을 요금제와 결합해 최종 핵심 타겟을 찾았습니다.</p>
      <div class="grid-3">
        <div class="insight"><span class="eyebrow">Largest tenure segment</span><b>{context["largest_tenure"]}</b><p>Top 10% 내 최대 가입기간 구간.</p></div>
        <div class="insight"><span class="eyebrow">Focus target</span><b>{context["focus_customers"]}</b><p>성장기/장기 Basic 고객 규모.</p></div>
        <div class="insight"><span class="eyebrow">Focus churn rate</span><b>{context["focus_churn"]}</b><p>핵심 타겟의 실제 이탈률.</p></div>
      </div>
      <div class="reading-guide">
        <strong>Segmentation logic</strong>
        가입기간별 규모와 이탈률을 먼저 확인한 뒤, 요금제 조건을 결합해 실행 가능한 핵심 타겟을 도출했습니다.
      </div>
      <div class="grid-2">
        <div>
          <div class="table-wrap">{context["high_tenure_table"]}</div>
        </div>
        <div class="chart"><img alt="Top 10 tenure profile" src="data:image/png;base64,{context["tenure_chart"]}" /></div>
      </div>
      <h3>가입기간 x 요금제</h3>
      <p class="caption">가입기간 x 요금제 조합은 고객 수와 churn rate를 함께 기준으로 정렬했습니다.</p>
      <div class="table-wrap">{context["tenure_subscription_table"]}</div>
    </section>
    <section>
      <h2>3. 결과 - 타겟 행동 특성 검증</h2>
      <p class="section-lead">핵심 타겟은 Basic 고객이라는 단일 조건으로 고른 것이 아닙니다. 최근 접속 공백, 낮은 시청시간, 낮은 완료율, 낮은 추천 클릭률이 함께 관찰됩니다.</p>
      <div class="grid-3">
        <div class="insight"><span class="eyebrow">Login gap</span><b>{context["focus_login_gap"]}</b><p>핵심 타겟 평균 최근 미접속 기간.</p></div>
        <div class="insight"><span class="eyebrow">Weekly watch</span><b>{context["focus_watch_min"]}</b><p>핵심 타겟 평균 주간 시청 시간.</p></div>
        <div class="insight"><span class="eyebrow">Recommendation response</span><b>{context["focus_rec_click"]}</b><p>핵심 타겟 평균 추천 클릭률.</p></div>
      </div>
      <div class="reading-guide">
        <strong>Behavior profile</strong>
        전체 test set, Top 10%, Focus target을 비교해 타겟의 행동 저하 패턴을 확인했습니다.
      </div>
      <div class="table-wrap">{context["feature_profile_table"]}</div>
      <div class="chart"><img alt="Focus target feature profile" src="data:image/png;base64,{context["focus_profile_chart"]}" /></div>
    </section>
    <section>
      <h2>4. 결과 - KMeans Cluster 결합 검증</h2>
      <p class="section-lead">Risk group은 모델 예측확률 기반 실행 구간이고, KMeans cluster는 train 데이터에서 fit한 비지도 군집입니다. 두 기준의 결합을 통해 타겟의 행동 패턴 일관성을 검토했습니다.</p>
      <div class="decision">
        <strong>High risk 고객의 {context["main_cluster_share"]}가 KMeans Cluster {context["main_cluster"]}에 집중</strong>
        모델 score가 높은 고객군이 비지도 군집에서도 같은 행동 패턴 집단으로 모여 있어, 최종 타겟 선정의 설명력이 강화됩니다.
      </div>
      <div class="reading-guide">
        <strong>해석 기준</strong>
        risk group은 캠페인 우선순위 기준, KMeans cluster는 행동 패턴 기준입니다.
        두 기준의 정합성은 타겟 선정의 보조 근거로 활용됩니다.
      </div>
      <div class="grid-2">
        <div class="table-wrap">{context["cluster_risk_table"]}</div>
        <div class="chart"><img alt="KMeans cluster by risk group" src="data:image/png;base64,{context["cluster_risk_heatmap"]}" /></div>
      </div>
    </section>
    <section>
      <h2>5. 최종 - 요금제 x 기기 인사이트</h2>
      <p class="section-lead">Basic + Mobile 고객 구간이 크기 때문에 최종 전략은 가격 민감도와 모바일 사용 맥락을 동시에 반영해야 합니다.</p>
      <div class="grid-3">
        <div class="insight"><span class="eyebrow">Largest plan-device</span><b>{context["largest_plan_device"]}</b><p>상위 10% 안에서 고객 수가 가장 큰 요금제 x 기기 조합입니다.</p></div>
        <div class="insight"><span class="eyebrow">Business implication</span><b>저가 방어</b><p>가격 민감 고객에게 완전 해지 대신 낮은 비용의 유지 선택지를 제공합니다.</p></div>
        <div class="insight"><span class="eyebrow">Strategy link</span><b>Mobile-only</b><p>모바일 사용 맥락을 반영한 전용 저가 요금제로 연결됩니다.</p></div>
      </div>
      <div class="reading-guide">
        <strong>Plan-device profile</strong>
        요금제와 기기 조합별 규모, 이탈률, 사용 저하 지표를 함께 비교했습니다.
      </div>
      <div class="table-wrap">{context["plan_device_table"]}</div>
      <h3>논문 기반 proxy 사후 점검</h3>
      <p class="caption">논문 기반 proxy는 최종 모델 입력이나 타겟 분할 기준이 아니라, Mobile-only 저가 요금제 제안의 해석 보조 근거로만 사용했습니다.</p>
      <div class="table-wrap">{context["paper_proxy_table"]}</div>
    </section>
    <footer>Source: outputs/churn_risk_scores_test.csv</footer>
  </main>
</body>
</html>
"""


def render_strategy(context: dict[str, object]) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Final Retention Strategy</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <nav>
      <a href="index.html">Report Home</a>
      <a href="eda_analysis.html">EDA Evidence</a>
      <a href="interactive_report.html">Interactive Report</a>
      <a href="modeling_methodology.html">Modeling Methodology</a>
      <a href="retention_analysis.html">Retention Analysis</a>
      <a class="active" href="top_risk_retention_strategy.html">Final Strategy</a>
    </nav>
    <div class="hero">
      <div>
        <h1>Final Retention Strategy</h1>
        <p class="subtitle">Retention 분석 결과를 바탕으로 가정, 실행 후보, 근거, 최종 운영 원칙 순서로 정리한 최종 전략 페이지입니다. 최종 타겟은 상위 10% 중 성장기/장기 Basic 고객입니다.</p>
      </div>
      <div class="kpi-row">
        <div class="kpi"><span>Focus target</span><b>{context["focus_customers"]}</b></div>
        <div class="kpi"><span>Focus target churn</span><b>{context["focus_churn"]}</b></div>
        <div class="kpi"><span>Top 10% share</span><b>{context["focus_share"]}</b></div>
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>1. 가정 - 모든 고위험 고객이 아니라 실행 가능한 핵심 타겟을 고른다</h2>
      <div class="decision">
        <strong>상위 10% risk 고객 중 성장기/장기 Basic 고객</strong>
        이 구간은 {context["focus_customers"]}으로 상위 10% 고객의 {context["focus_share"]}를 차지하고, 실제 churn rate는 {context["focus_churn"]}입니다.
        제한된 예산에서 가장 먼저 보호해야 할 고객군입니다.
      </div>
      <div class="reading-guide">
        <strong>Target boundary</strong>
        Top 10%는 threshold로 분류한 집단이 아니라, 최종 모델 <code>{context["final_model"]}</code>의 <code>churn_probability</code>를 내림차순 정렬했을 때 전체 test 고객 {context["total_customers"]}명 중 상위 {context["top10_customers"]}명입니다.
        이 중 성장기/장기 Basic 고객 {context["focus_customers"]}을 최종 focus target으로 좁혔기 때문에, Top 10% 전체 churn rate({context["top10_churn"]})와 focus target churn rate({context["focus_churn"]})는 서로 다른 모수에서 계산됩니다.
      </div>
      <div class="grid-3">
        <div class="evidence">
          <span class="eyebrow">Why not Top 30%</span>
          <strong>검증 범위와 실행 범위를 분리</strong>
          <p>Top 30%는 이탈자의 {context["top30_captured"]}를 포착하지만, 유료 혜택 집행 범위로는 넓습니다.</p>
        </div>
        <div class="evidence">
          <span class="eyebrow">Why Top 10%</span>
          <strong>가장 높은 위험도</strong>
          <p>Top 10%의 실제 churn rate는 {context["top10_churn"]}로, 초기 캠페인 효율을 검증하기 좋습니다.</p>
        </div>
        <div class="evidence">
          <span class="eyebrow">Why Basic</span>
          <strong>규모와 이탈률 동시 충족</strong>
          <p>성장기/장기 Basic은 고객 수 기준 핵심 구간이며 가격 민감도 대응이 가능합니다.</p>
        </div>
      </div>
    </section>
    <section>
      <h2>2. 실험 - 실행 액션 후보 설계</h2>
      <div class="grid-2">
        <div class="action">
          <span class="eyebrow">Action 1</span>
          <strong>중/장기 Basic 업그레이드 쿠폰</strong>
          <p>상위 risk 고객에게만 Standard 1개월 업그레이드 쿠폰을 제공합니다. 신규 고객 확보용 프로모션을 기존 고위험 고객 retention 용도로 재해석한 전략입니다.</p>
          <ul>
            <li>대상: 성장기/장기 Basic 중 risk score 상위 고객</li>
            <li>혜택: Standard 1개월 업그레이드</li>
            <li>종료 후: 원래 요금제 복귀 또는 상위 요금제 유지 선택</li>
          </ul>
        </div>
        <div class="action">
          <span class="eyebrow">Action 2</span>
          <strong>Mobile-only 저가 요금제</strong>
          <p>모바일에서만 시청 가능한 저가 요금제를 신설해 가격 민감 고객의 완전 해지를 낮은 비용의 유지로 전환합니다. 실제 가격 민감도는 직접 관측되지 않으므로, 사용량 대비 요금제 가치 proxy와 Basic + Mobile 집중도를 함께 해석했습니다.</p>
          <ul>
            <li>대상: Mobile 사용 비중이 높은 상위 risk 고객</li>
            <li>제약: 모바일 앱 전용, 동시접속/화질/TV 이용 제한</li>
            <li>목표: 해지 대신 저가 유지 플랜으로 전환</li>
          </ul>
        </div>
      </div>
    </section>
    <section>
      <h2>3. 근거 - Benchmark Logic</h2>
      <p class="section-lead">
        Netflix는 과거 호주와 인도에서 신규 가입자를 대상으로 30일 무료 체험 대신 한시적 요금제 업그레이드 프로모션을 운영한 사례가 있습니다.
        본 프로젝트는 같은 업그레이드 메커니즘을 신규 유입이 아니라 기존 중/장기 Basic 고객 이탈 방지에 적용합니다.
      </p>
      <p class="small">
        References:
        <a href="https://www.techradar.com/news/netflix-has-dropped-its-30-day-free-trial-period-in-australia">TechRadar Australia case</a>,
        <a href="https://www.gadgets360.com/entertainment/news/netflix-free-upgrade-plans-standard-premium-subscription-offer-price-india-30-days-2234597/amp">Gadgets360 India case</a>,
        <a href="https://help.netflix.com/en/node/16282">Netflix Help Center</a>
      </p>
    </section>
    <section>
      <h2>4. 결과 - 비용 구조와 리스크</h2>
      <div class="reading-guide">
        <strong>요금제 가격 가정</strong>
        Basic 9,500원/월 · Standard 12,000원/월 · Premium 13,500원/월
      </div>
      <div class="grid-2">
        <div class="evidence">
          <span class="eyebrow">업그레이드 쿠폰</span>
          <strong>현금성 할인 없이 상위 경험 제공</strong>
          <p>고객은 Basic 요금을 유지한 채 1개월간 Standard 혜택을 체험합니다. 현금 지출은 제한적이지만 동시접속, 화질, 트래픽 증가에 따른 인프라 한계비용은 존재합니다.</p>
        </div>
        <div class="evidence">
          <span class="eyebrow">운영 리스크</span>
          <strong>쿠폰 종료 후 행동 관리 필요</strong>
          <p>프로모션 종료 시 원래 요금제로 복귀하거나 상위 요금제를 유지하도록 선택지를 명확히 제시해야 합니다. 다운그레이드 경험이 불만으로 이어지는지 별도 모니터링이 필요합니다.</p>
        </div>
      </div>
    </section>
    <section>
      <h2>5. 실험 - A/B 테스트 설계</h2>
      <div class="grid-3">
        <div class="card"><span class="eyebrow">Population</span><strong>Focus target {context["focus_customers"]}</strong><p>상위 10% 중 성장기/장기 Basic 고객을 control과 treatment로 무작위 분할합니다.</p></div>
        <div class="card"><span class="eyebrow">Primary metric</span><strong>30일 잔존율</strong><p>보조 지표는 60일 잔존율, Standard 유지율, Mobile-only 전환율, 메시지 수신 후 접속 회복률입니다.</p></div>
        <div class="card"><span class="eyebrow">Guardrail</span><strong>ARPU와 해지 가속</strong><p>쿠폰 종료 후 즉시 해지, 다운그레이드 불만, 저가 요금제 전환에 따른 매출 희석을 함께 관찰합니다.</p></div>
      </div>
    </section>
    <section>
      <h2>6. 최종 - 캠페인 운영 원칙</h2>
      <div class="grid-3">
        <div class="card"><span class="eyebrow">Priority</span><strong>Risk score first</strong><p>캠페인 대상 여부는 모델 ranking으로 결정합니다.</p></div>
        <div class="card"><span class="eyebrow">Segmentation</span><strong>KMeans cluster second</strong><p>메시지와 혜택 세분화에는 KMeans cluster를 보조 기준으로 사용합니다.</p></div>
        <div class="card"><span class="eyebrow">Budget control</span><strong>Top 10% pilot</strong><p>초기 집행은 상위 10%로 제한하고 효과 검증 후 확장합니다.</p></div>
      </div>
    </section>
    <footer>Detailed evidence is available in <a href="retention_analysis.html">Retention Analysis</a>.</footer>
  </main>
</body>
</html>
"""


def main() -> None:
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RISK_SCORE_PATH).sort_values("churn_probability", ascending=False).reset_index(drop=True)
    df = add_segments(df)
    plan_price = {"Basic": 9500, "Standard": 12000, "Premium": 13500}
    df["plan_price"] = df["subscription_type"].map(plan_price)
    value_cols = [
        "avg_watch_time_minutes_per_week",
        "watch_sessions_per_week",
        "completion_rate",
        "recommendation_click_rate",
        "app_rating",
    ]
    normalized = pd.DataFrame(index=df.index)
    for col in value_cols:
        span = df[col].max() - df[col].min()
        normalized[col] = 0 if span == 0 else (df[col] - df[col].min()) / span * 100
    df["usage_value_score"] = normalized.mean(axis=1)
    df["value_for_money_score"] = df["usage_value_score"] / (df["plan_price"] / plan_price["Basic"])
    df["price_burden_proxy"] = (100 - df["value_for_money_score"]).clip(lower=0, upper=100)
    topk = build_topk(df)
    high = df[df["risk_group"] == "High: top 10%"].copy()
    focus = high[
        high["tenure_segment"].isin(["Growing 13-36m", "Long 37-72m"])
        & (high["subscription_type"] == "Basic")
    ].copy()

    high_tenure = (
        high.groupby("tenure_segment", observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
            basic_share=("subscription_type", lambda s: (s == "Basic").mean()),
            mobile_share=("primary_device", lambda s: (s == "Mobile").mean()),
        )
        .reset_index()
    )
    tenure_subscription = (
        high.groupby(["tenure_segment", "subscription_type"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
            avg_last_login_days=("days_since_last_login", "mean"),
            avg_weekly_watch_minutes=("avg_watch_time_minutes_per_week", "mean"),
        )
        .reset_index()
        .query("customers >= 20")
        .sort_values(["customers", "churn_rate"], ascending=[False, False])
        .head(12)
    )
    feature_cols = [
        "days_since_last_login",
        "avg_watch_time_minutes_per_week",
        "watch_sessions_per_week",
        "completion_rate",
        "recommendation_click_rate",
        "app_rating",
    ]
    profile_rows = []
    for name, source in [("All test", df), ("Top 10%", high), ("Focus target", focus)]:
        row = {
            "group": name,
            "customers": len(source),
            "churn_rate": source["actual_churn"].mean(),
            "basic_share": (source["subscription_type"] == "Basic").mean(),
            "mobile_share": (source["primary_device"] == "Mobile").mean(),
        }
        row.update(source[feature_cols].mean().to_dict())
        profile_rows.append(row)
    feature_profile = pd.DataFrame(profile_rows)

    cluster_risk = (
        df.groupby(["cluster", "risk_group"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
            basic_share=("subscription_type", lambda s: (s == "Basic").mean()),
            mobile_share=("primary_device", lambda s: (s == "Mobile").mean()),
        )
        .reset_index()
        .sort_values(["risk_group", "customers"], ascending=[True, False])
    )
    high_cluster_counts = cluster_risk[cluster_risk["risk_group"] == "High: top 10%"].sort_values("customers", ascending=False)
    main_high_cluster = high_cluster_counts.iloc[0]

    plan_device = (
        high.groupby(["subscription_type", "primary_device"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_last_login_days=("days_since_last_login", "mean"),
            avg_weekly_watch_minutes=("avg_watch_time_minutes_per_week", "mean"),
        )
        .reset_index()
        .query("customers >= 15")
        .sort_values(["customers", "churn_rate"], ascending=[False, False])
        .head(10)
    )
    proxy_rows = []
    for name, source in [
        ("All test", df),
        ("Top 10%", high),
        ("Focus target", focus),
        ("Top 10% Basic + Mobile", high[(high["subscription_type"] == "Basic") & (high["primary_device"] == "Mobile")]),
    ]:
        proxy_rows.append(
            {
                "group": name,
                "customers": len(source),
                "churn_rate": source["actual_churn"].mean(),
                "mobile_share": (source["primary_device"] == "Mobile").mean(),
                "avg_usage_value": source["usage_value_score"].mean(),
                "avg_value_for_money": source["value_for_money_score"].mean(),
                "price_burden_proxy": source["price_burden_proxy"].mean(),
            }
        )
    paper_proxy = pd.DataFrame(proxy_rows)

    top10_row = topk.loc[topk["target_pct"] == 0.10].iloc[0]
    top30_row = topk.loc[topk["target_pct"] == 0.30].iloc[0]
    focus_share = len(focus) / len(high)
    largest_tenure = high_tenure.sort_values("customers", ascending=False).iloc[0]
    largest_plan_device = plan_device.sort_values("customers", ascending=False).iloc[0]

    def fmt_table(
        source: pd.DataFrame,
        pct_cols: list[str],
        num_cols: list[str] | None = None,
        highlight_terms: tuple[str, ...] = (),
    ) -> str:
        table = source.copy()
        for col in pct_cols:
            table[col] = table[col].map(pct)
        for col in num_cols or []:
            table[col] = table[col].map(lambda v: num(v, 1))
        return dataframe_to_html(table, highlight_terms)

    topk_table = topk.assign(
        target_pct=topk["target_pct"].map(lambda v: f"Top {int(v * 100)}%"),
        churn_rate=topk["churn_rate"].map(pct),
        lift=topk["lift"].map(lambda v: f"{v:.2f}x"),
        captured_churner=topk["captured_churner"].map(pct),
    ).rename(
        columns={
            "target_pct": "Target",
            "customers": "Customers",
            "churn_rate": "Actual churn rate",
            "lift": "Lift",
            "captured_churner": "Captured churners",
        }
    )
    high_tenure_table = fmt_table(
        high_tenure.rename(
            columns={
                "tenure_segment": "Tenure segment",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "avg_probability": "Avg risk score",
                "basic_share": "Basic share",
                "mobile_share": "Mobile share",
            }
        ),
        ["Actual churn rate", "Avg risk score", "Basic share", "Mobile share"],
        highlight_terms=("Growing 13-36m", "Long 37-72m"),
    )
    tenure_subscription_table = fmt_table(
        tenure_subscription.rename(
            columns={
                "tenure_segment": "Tenure segment",
                "subscription_type": "Plan",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "avg_probability": "Avg risk score",
                "avg_last_login_days": "Avg days since login",
                "avg_weekly_watch_minutes": "Avg weekly watch min",
            }
        ),
        ["Actual churn rate", "Avg risk score"],
        ["Avg days since login", "Avg weekly watch min"],
        highlight_terms=("Growing 13-36m|Basic", "Long 37-72m|Basic"),
    )
    feature_profile_table = fmt_table(
        feature_profile.rename(
            columns={
                "group": "Group",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "basic_share": "Basic share",
                "mobile_share": "Mobile share",
                "days_since_last_login": "Avg days since login",
                "avg_watch_time_minutes_per_week": "Avg weekly watch min",
                "watch_sessions_per_week": "Avg weekly sessions",
                "completion_rate": "Avg completion",
                "recommendation_click_rate": "Avg rec click",
                "app_rating": "Avg app rating",
            }
        ),
        ["Actual churn rate", "Basic share", "Mobile share"],
        ["Avg days since login", "Avg weekly watch min", "Avg weekly sessions", "Avg completion", "Avg rec click", "Avg app rating"],
        highlight_terms=("Focus target",),
    )
    cluster_risk_table = fmt_table(
        cluster_risk.assign(cluster=cluster_risk["cluster"].map(lambda v: f"Cluster {v}")).rename(
            columns={
                "cluster": "KMeans cluster",
                "risk_group": "Risk group",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "avg_probability": "Avg risk score",
                "basic_share": "Basic share",
                "mobile_share": "Mobile share",
            }
        ),
        ["Actual churn rate", "Avg risk score", "Basic share", "Mobile share"],
        highlight_terms=("Cluster 1|High: top 10%",),
    )
    plan_device_table = fmt_table(
        plan_device.rename(
            columns={
                "subscription_type": "Plan",
                "primary_device": "Device",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "avg_last_login_days": "Avg days since login",
                "avg_weekly_watch_minutes": "Avg weekly watch min",
            }
        ),
        ["Actual churn rate"],
        ["Avg days since login", "Avg weekly watch min"],
        highlight_terms=("Basic|Mobile",),
    )
    paper_proxy_table = fmt_table(
        paper_proxy.rename(
            columns={
                "group": "Group",
                "customers": "Customers",
                "churn_rate": "Actual churn rate",
                "mobile_share": "Mobile share",
                "avg_usage_value": "Avg usage value",
                "avg_value_for_money": "Avg value for money",
                "price_burden_proxy": "Price burden proxy",
            }
        ),
        ["Actual churn rate", "Mobile share"],
        ["Avg usage value", "Avg value for money", "Price burden proxy"],
        highlight_terms=("Focus target", "Top 10% Basic + Mobile"),
    )

    context = {
        "final_model": model_label(str(df["final_model"].mode().iloc[0])) if "final_model" in df.columns else "Stacking (original features)",
        "top30_captured": pct(top30_row["captured_churner"]),
        "top10_churn": pct(top10_row["churn_rate"]),
        "total_customers": f"{len(df):,}",
        "total_churners": f"{int(df['actual_churn'].sum()):,}",
        "top10_customers": f"{int(len(df) * 0.10):,}",
        "top20_customers": f"{int(len(df) * 0.20):,}",
        "top30_customers": f"{int(len(df) * 0.30):,}",
        "focus_customers": f"{len(focus):,}명",
        "focus_churn": pct(focus["actual_churn"].mean()),
        "focus_share": pct(focus_share),
        "largest_tenure": str(largest_tenure["tenure_segment"]).replace(" 13-36m", "").replace(" 37-72m", ""),
        "focus_login_gap": f"{focus['days_since_last_login'].mean():.1f}일",
        "focus_watch_min": f"{focus['avg_watch_time_minutes_per_week'].mean():.1f}분",
        "focus_rec_click": f"{focus['recommendation_click_rate'].mean():.1f}%",
        "main_cluster": main_high_cluster["cluster"],
        "main_cluster_share": pct(main_high_cluster["customers"] / len(high)),
        "largest_plan_device": f"{largest_plan_device['subscription_type']} + {largest_plan_device['primary_device']}",
        "topk_table": dataframe_to_html(topk_table, ("Top 10%", "Top 30%")),
        "high_tenure_table": high_tenure_table,
        "tenure_subscription_table": tenure_subscription_table,
        "feature_profile_table": feature_profile_table,
        "cluster_risk_table": cluster_risk_table,
        "plan_device_table": plan_device_table,
        "paper_proxy_table": paper_proxy_table,
        "topk_chart": make_topk_chart(topk),
        "tenure_chart": make_tenure_chart(high_tenure),
        "focus_profile_chart": make_focus_profile_chart(feature_profile),
        "cluster_risk_heatmap": make_cluster_risk_heatmap(cluster_risk),
    }

    ANALYSIS_PATH.write_text(render_analysis(context), encoding="utf-8")
    STRATEGY_PATH.write_text(render_strategy(context), encoding="utf-8")
    print(ANALYSIS_PATH)
    print(STRATEGY_PATH)


if __name__ == "__main__":
    main()
