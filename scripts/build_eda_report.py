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

DATA_PATH = ROOT / "data" / "user_behavior_50000" / "netflix_user_behavior_churn_50000v2.csv"
REPORT_PATH = ROOT / "reports" / "eda_analysis.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


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
        for term in highlight_terms:
            if term in row_html:
                return row_html.replace("<tr>", '<tr class="highlight-row">', 1)
        return row_html

    return re.sub(r"<tr>\s*(?:<td>.*?</td>\s*)+</tr>", mark_row, html, flags=re.S)


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_segment"] = pd.cut(
        df["account_age_months"],
        bins=[0, 12, 36, 72, 10**9],
        labels=["New <=12m", "Growing 13-36m", "Long 37-72m", "Very long 73m+"],
        include_lowest=True,
    )
    df["login_segment"] = pd.cut(
        df["days_since_last_login"],
        bins=[-1, 7, 30, 10**9],
        labels=["Active <=7d", "Watch 8-30d", "Dormant 31d+"],
    )
    return df


def make_behavior_chart(summary: pd.DataFrame) -> str:
    metrics = [
        ("days_since_last_login", "Days since login"),
        ("avg_watch_time_minutes_per_week", "Weekly watch minutes"),
        ("completion_rate", "Completion rate"),
        ("recommendation_click_rate", "Recommendation click rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.0))
    axes = axes.ravel()
    colors = ["#287a47", "#b42318"]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        bars = ax.bar(summary["Group"], summary[metric], color=colors)
        ax.set_title(title)
        ymax = max(summary[metric]) or 1
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + ymax * 0.025, f"{height:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_segment_chart(segment: pd.DataFrame, title: str, label_col: str) -> str:
    labels = segment[label_col].astype(str)
    fig, ax1 = plt.subplots(figsize=(10.2, 4.1))
    ax2 = ax1.twinx()
    ax1.bar(labels, segment["customers"], color="#46566f", alpha=0.86)
    ax2.plot(labels, segment["churn_rate"] * 100, color="#b42318", marker="o", linewidth=2.4)
    ax1.set_title(title)
    ax1.set_ylabel("Customers")
    ax2.set_ylabel("Churn rate (%)")
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=12)
    for i, value in enumerate(segment["churn_rate"] * 100):
        ax2.text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9, color="#8a1c13")
    fig.tight_layout()
    return fig_to_base64(fig)


def css() -> str:
    return """
    :root { --text:#121826; --muted:#5e6b7a; --line:#dbe2ea; --blue:#1f5fbf; --red:#b42318; --soft:#f6f8fb; }
    * { box-sizing: border-box; }
    body { margin:0; color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.56; background:#fff; }
    header { background:#f7f9fc; border-bottom:1px solid var(--line); padding:34px 32px 30px; }
    nav { max-width:1120px; margin:0 auto 22px; display:flex; flex-wrap:wrap; gap:8px; }
    nav a { color:#334155; text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:7px 12px; background:#fff; font-size:14px; }
    nav a.active { color:#fff; background:var(--blue); border-color:var(--blue); }
    .hero { max-width:1120px; margin:0 auto; display:grid; grid-template-columns:1.4fr .9fr; gap:20px; align-items:end; }
    h1 { margin:0; font-size:36px; line-height:1.18; letter-spacing:0; }
    .subtitle { margin:12px 0 0; color:var(--muted); font-size:16px; max-width:760px; }
    .kpi-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
    .kpi, .insight, .box { border:1px solid var(--line); border-radius:8px; background:#fff; padding:15px; }
    .kpi span, .eyebrow { display:block; color:var(--muted); font-size:13px; font-weight:650; }
    .kpi b { display:block; margin-top:4px; font-size:24px; line-height:1.18; overflow-wrap:anywhere; }
    main { max-width:1120px; margin:0 auto; padding:30px 28px 64px; }
    section { margin:0 0 42px; }
    h2 { margin:0 0 14px; font-size:24px; letter-spacing:0; }
    h3 { margin:22px 0 8px; font-size:18px; }
    p { margin:8px 0 12px; }
    .section-lead { max-width:850px; color:var(--muted); }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .insight b, .box strong { display:block; margin:4px 0 6px; font-size:20px; line-height:1.25; }
    .insight p, .box p { color:var(--muted); font-size:14px; margin-bottom:0; }
    .reading-guide { border-left:4px solid var(--blue); background:#f0f6ff; border-radius:8px; padding:14px 16px; margin:14px 0 16px; }
    .reading-guide strong { display:block; color:#174ea6; margin-bottom:4px; }
    .chart { border:1px solid var(--line); border-radius:8px; background:#fff; padding:12px; margin:14px 0 20px; }
    .chart img { display:block; width:100%; height:auto; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:12px 0 22px; }
    .data-table { width:100%; border-collapse:collapse; font-size:14px; background:#fff; }
    .data-table th { text-align:left; padding:10px 9px; border-bottom:2px solid #334155; white-space:nowrap; background:#fbfcfe; }
    .data-table td { padding:9px; border-bottom:1px solid var(--line); vertical-align:top; }
    .data-table tr.highlight-row td { background:#fff7ed; font-weight:650; }
    .data-table tbody tr:hover td { background:#f8fafc; }
    footer { margin-top:40px; border-top:1px solid var(--line); padding-top:18px; color:var(--muted); font-size:13px; }
    @media (max-width:860px) { header{padding:28px 20px;} main{padding:24px 18px 48px;} .hero,.grid-2,.grid-3,.kpi-row{grid-template-columns:1fr;} h1{font-size:29px;} }
    """


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = add_segments(pd.read_csv(DATA_PATH))

    behavior_cols = [
        "days_since_last_login",
        "avg_watch_time_minutes_per_week",
        "watch_sessions_per_week",
        "completion_rate",
        "recommendation_click_rate",
        "app_rating",
    ]
    behavior_summary = df.groupby("churned")[behavior_cols].mean().reset_index()
    behavior_summary["Group"] = behavior_summary["churned"].map({0: "Retained", 1: "Churned"})
    behavior_display = behavior_summary[["Group", *behavior_cols]].rename(
        columns={
            "days_since_last_login": "Avg days since login",
            "avg_watch_time_minutes_per_week": "Avg weekly watch min",
            "watch_sessions_per_week": "Avg weekly sessions",
            "completion_rate": "Avg completion",
            "recommendation_click_rate": "Avg rec click",
            "app_rating": "Avg app rating",
        }
    )
    for col in behavior_display.columns[1:]:
        behavior_display[col] = behavior_display[col].map(lambda v: num(v, 1))

    corr = df[behavior_cols + ["churned"]].corr(numeric_only=True)["churned"].drop("churned").abs().sort_values(ascending=False)
    top_corr = corr.head(5)
    signal_labels = {
        "days_since_last_login": "Login recency",
        "avg_watch_time_minutes_per_week": "Weekly watch time",
        "watch_sessions_per_week": "Weekly sessions",
        "completion_rate": "Completion rate",
        "recommendation_click_rate": "Recommendation click",
        "app_rating": "App rating",
    }
    strongest_signal = signal_labels.get(str(top_corr.index[0]), str(top_corr.index[0]))

    plan = (
        df.groupby("subscription_type", observed=True)
        .agg(customers=("user_id", "size"), churn_rate=("churned", "mean"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    device = (
        df.groupby("primary_device", observed=True)
        .agg(customers=("user_id", "size"), churn_rate=("churned", "mean"))
        .reset_index()
        .sort_values("customers", ascending=False)
    )
    tenure = (
        df.groupby("tenure_segment", observed=True)
        .agg(customers=("user_id", "size"), churn_rate=("churned", "mean"))
        .reset_index()
        .sort_values("customers", ascending=False)
    )
    plan_device = (
        df.groupby(["subscription_type", "primary_device"], observed=True)
        .agg(customers=("user_id", "size"), churn_rate=("churned", "mean"))
        .reset_index()
        .sort_values(["customers", "churn_rate"], ascending=[False, False])
        .head(10)
    )

    def fmt_table(source: pd.DataFrame, pct_cols: list[str], highlight_terms: tuple[str, ...] = ()) -> str:
        table = source.copy()
        for col in pct_cols:
            table[col] = table[col].map(pct)
        return dataframe_to_html(table, highlight_terms)

    plan_table = fmt_table(plan.rename(columns={"subscription_type": "Plan", "customers": "Customers", "churn_rate": "Churn rate"}), ["Churn rate"], ("Basic",))
    device_table = fmt_table(device.rename(columns={"primary_device": "Device", "customers": "Customers", "churn_rate": "Churn rate"}), ["Churn rate"], ("Mobile",))
    tenure_table = fmt_table(tenure.rename(columns={"tenure_segment": "Tenure segment", "customers": "Customers", "churn_rate": "Churn rate"}), ["Churn rate"], ("Growing", "Long"))
    plan_device_table = fmt_table(
        plan_device.rename(columns={"subscription_type": "Plan", "primary_device": "Device", "customers": "Customers", "churn_rate": "Churn rate"}),
        ["Churn rate"],
        ("Basic",),
    )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EDA Evidence</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <nav>
      <a href="index.html">Report Home</a>
      <a class="active" href="eda_analysis.html">EDA Evidence</a>
      <a href="interactive_report.html">Interactive Report</a>
      <a href="modeling_methodology.html">Modeling Methodology</a>
      <a href="retention_analysis.html">Retention Analysis</a>
      <a href="top_risk_retention_strategy.html">Final Strategy</a>
    </nav>
    <div class="hero">
      <div>
        <h1>EDA Evidence</h1>
        <p class="subtitle">01 EDA 노트북의 가정, 실험, 결과, 최종 질문 구조를 유지해 정리한 페이지입니다. 이 단계에서는 최종 전략을 확정하지 않고, churn과 연결되는 데이터 신호와 후속 모델링 질문을 정리합니다.</p>
      </div>
      <div class="kpi-row">
        <div class="kpi"><span>Customers</span><b>{len(df):,}</b></div>
        <div class="kpi"><span>Churn rate</span><b>{pct(df["churned"].mean())}</b></div>
        <div class="kpi"><span>Strongest signal</span><b>{strongest_signal}</b></div>
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>1. 가정 - churn과 연결된 행동 신호가 존재한다</h2>
      <p class="section-lead">분석 데이터는 50,000명의 synthetic Netflix-style 고객 행동 데이터이며, churn 고객은 전체의 {pct(df["churned"].mean())}입니다. EDA의 목적은 특정 전략을 바로 선택하는 것이 아니라, 어떤 행동/상품 신호가 churn과 함께 움직이는지 확인하는 것입니다.</p>
      <div class="grid-3">
        <div class="insight"><span class="eyebrow">Numeric behavior</span><b>{len(behavior_cols)}개</b><p>시청량, 세션, 완료율, 추천 클릭률, 최근 접속 공백 등.</p></div>
        <div class="insight"><span class="eyebrow">Categorical context</span><b>상품/환경</b><p>요금제, 결제수단, 기기, 장르, 유입 경로, 이용 시간대.</p></div>
        <div class="insight"><span class="eyebrow">EDA role</span><b>관찰과 질문 정리</b><p>어떤 피처가 churn과 연결되는지 확인하고 모델링에서 검증할 후보 신호를 좁혔습니다.</p></div>
      </div>
    </section>
    <section>
      <h2>2. 실험 - 타겟 분포와 행동 피처 차이 확인</h2>
      <p class="section-lead">이탈 고객은 최근 로그인 공백이 길고, 시청 시간과 세션 수, 완료율, 추천 클릭률, 앱 평점이 낮은 방향으로 관찰됩니다.</p>
      <div class="table-wrap">{dataframe_to_html(behavior_display, ("Churned",))}</div>
      <div class="chart"><img alt="Behavior difference by churn" src="data:image/png;base64,{make_behavior_chart(behavior_summary)}" /></div>
      <div class="reading-guide">
        <strong>모델링 연결</strong>
        이 단계에서는 최근성, 사용량, 콘텐츠 반응이 churn과 함께 움직인다는 관찰을 얻었습니다. 이 관찰은 다음 모델링 페이지에서 원본 행동 피처를 중심으로 실험하는 근거가 됩니다.
      </div>
    </section>
    <section>
      <h2>3. 결과 - 요금제, 기기, 가입기간별 패턴</h2>
      <p class="section-lead">EDA 단계에서는 단일 피처뿐 아니라 상품/사용 환경별 churn 차이를 확인했습니다. 여기서는 특정 액션을 확정하지 않고, 요금제와 기기 정보가 후속 세그먼트 분석에서 검토할 만한 변수인지 확인합니다.</p>
      <div class="grid-2">
        <div>
          <h3>요금제별 churn</h3>
          <div class="table-wrap">{plan_table}</div>
        </div>
        <div>
          <h3>기기별 churn</h3>
          <div class="table-wrap">{device_table}</div>
        </div>
      </div>
      <div class="chart"><img alt="Plan churn profile" src="data:image/png;base64,{make_segment_chart(plan.rename(columns={"subscription_type": "segment"}), "Churn by Plan", "segment")}" /></div>
      <h3>가입기간별 churn</h3>
      <div class="table-wrap">{tenure_table}</div>
      <div class="chart"><img alt="Tenure churn profile" src="data:image/png;base64,{make_segment_chart(tenure.rename(columns={"tenure_segment": "segment"}), "Churn by Tenure Segment", "segment")}" /></div>
    </section>
    <section>
      <h2>4. 결과 - 교차 세그먼트 관찰</h2>
      <p class="section-lead">요금제와 기기 조합을 함께 보면 일부 조합에서 규모와 churn rate 차이가 함께 관찰됩니다. 이 결과는 이후 risk score 기반 세그먼트 분석에서 다시 검증할 후보입니다.</p>
      <div class="table-wrap">{plan_device_table}</div>
    </section>
    <section>
      <h2>5. 최종 - 모델링으로 넘길 질문</h2>
      <div class="grid-3">
        <div class="box"><span class="eyebrow">Feature question</span><strong>어떤 신호를 모델에 넣을 것인가</strong><p>최근 접속, 시청량, 세션, 완료율, 추천 반응을 우선 후보 피처로 넘깁니다.</p></div>
        <div class="box"><span class="eyebrow">Modeling question</span><strong>분류보다 순위화가 더 적절한가</strong><p>불균형 타겟에서 churn 고객을 상위 위험 구간에 얼마나 모을 수 있는지 모델링에서 검증합니다.</p></div>
        <div class="box"><span class="eyebrow">Segmentation question</span><strong>고위험 고객은 어떤 세그먼트인가</strong><p>요금제, 기기, 가입기간 차이가 risk ranking 상위 고객 안에서도 유지되는지 후속 분석에서 확인합니다.</p></div>
      </div>
    </section>
    <footer>Source: data/user_behavior_50000/netflix_user_behavior_churn_50000v2.csv, notebooks/01_eda_segment_insight.ipynb</footer>
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
