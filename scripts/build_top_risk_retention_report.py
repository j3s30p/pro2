from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

RISK_SCORE_PATH = ROOT / "outputs" / "churn_risk_scores_test.csv"
MODEL_RESULT_PATH = ROOT / "outputs" / "churn_model_final_result.csv"
THRESHOLD_RESULT_PATH = ROOT / "outputs" / "churn_threshold_result.csv"
REPORT_PATH = ROOT / "reports" / "top_risk_retention_strategy.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dataframe_to_html(df: pd.DataFrame, classes: str = "data-table") -> str:
    return df.to_html(index=False, classes=classes, border=0, escape=False)


def build_topk(df: pd.DataFrame) -> pd.DataFrame:
    base_rate = df["actual_churn"].mean()
    rows = []
    for target_pct in [0.05, 0.10, 0.20, 0.30]:
        n = int(len(df) * target_pct)
        top = df.head(n)
        churn_rate = top["actual_churn"].mean()
        captured = top["actual_churn"].sum() / df["actual_churn"].sum()
        rows.append(
            {
                "target_pct": target_pct,
                "customers": n,
                "churn_rate": churn_rate,
                "lift": churn_rate / base_rate,
                "captured_churner": captured,
            }
        )
    return pd.DataFrame(rows)


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


def make_topk_chart(topk: pd.DataFrame) -> str:
    labels = [f"Top {int(v * 100)}%" for v in topk["target_pct"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    axes[0].bar(labels, topk["churn_rate"] * 100, color="#2563eb")
    axes[0].set_title("Actual Churn Rate by Target Size")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(topk["churn_rate"] * 100):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    axes[1].bar(labels, topk["captured_churner"] * 100, color="#16a34a")
    axes[1].set_title("Captured Churners by Target Size")
    axes[1].set_ylabel("Captured churners (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(topk["captured_churner"] * 100):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_tenure_chart(high_tenure: pd.DataFrame) -> str:
    labels = high_tenure["tenure_segment"].astype(str).tolist()
    fig, ax1 = plt.subplots(figsize=(9.8, 4.2))
    ax2 = ax1.twinx()

    bars = ax1.bar(labels, high_tenure["customers"], color="#475569", alpha=0.82)
    line = ax2.plot(
        labels,
        high_tenure["churn_rate"] * 100,
        color="#dc2626",
        marker="o",
        linewidth=2.3,
        label="Churn rate",
    )

    ax1.set_title("Top 10% Risk Customers by Tenure Segment")
    ax1.set_ylabel("Customers")
    ax2.set_ylabel("Actual churn rate (%)")
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=15)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{int(h)}", ha="center", fontsize=9)
    for i, v in enumerate(high_tenure["churn_rate"] * 100):
        ax2.text(i, v + 2.2, f"{v:.1f}%", ha="center", color="#991b1b", fontsize=9)

    ax2.legend(line, ["Churn rate"], loc="upper right")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_risk_tenure_heatmap(risk_tenure: pd.DataFrame) -> str:
    pivot = risk_tenure.pivot(index="risk_group", columns="tenure_segment", values="churn_rate")
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    image = ax.imshow(pivot.values * 100, cmap="Reds", vmin=0, vmax=100)

    ax.set_title("Churn Rate Heatmap: Risk Group x Tenure")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index.astype(str))

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j] * 100:.1f}%", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Churn rate (%)")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_activity_chart(high_activity: pd.DataFrame) -> str:
    table = high_activity.pivot(index="tenure_segment", columns="activity_segment", values="customers").fillna(0)
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    table.plot(kind="bar", stacked=True, ax=ax, color=["#22c55e", "#f59e0b", "#ef4444"])
    ax.set_title("Top 10% Risk Customers: Tenure x Last Login Activity")
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Last login")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_focus_profile_chart(feature_profile: pd.DataFrame) -> str:
    metrics = [
        "days_since_last_login",
        "avg_watch_time_minutes_per_week",
        "completion_rate",
        "recommendation_click_rate",
    ]
    titles = ["Days since login", "Weekly watch min", "Completion rate", "Recommendation click rate"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    axes = axes.ravel()

    for ax, metric, title in zip(axes, metrics, titles, strict=True):
        bars = ax.bar(feature_profile["group"], feature_profile[metric], color=["#64748b", "#2563eb", "#dc2626"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(feature_profile[metric]) * 0.025, f"{h:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    return fig_to_base64(fig)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RISK_SCORE_PATH)
    df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    df = add_segments(df)

    base_churn_rate = df["actual_churn"].mean()
    topk = build_topk(df)
    high = df[df["risk_group"] == "High: top 10%"].copy()
    focus_mask = high["tenure_segment"].isin(["Growing 13-36m", "Long 37-72m"]) & (
        high["subscription_type"] == "Basic"
    )
    focus = high[focus_mask].copy()

    high_tenure = (
        high.groupby("tenure_segment", observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
            avg_last_login_days=("days_since_last_login", "mean"),
            avg_weekly_watch_minutes=("avg_watch_time_minutes_per_week", "mean"),
            basic_share=("subscription_type", lambda s: (s == "Basic").mean()),
            mobile_share=("primary_device", lambda s: (s == "Mobile").mean()),
        )
        .reset_index()
    )

    high_activity = (
        high.groupby(["tenure_segment", "activity_segment"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
        )
        .reset_index()
    )

    risk_tenure = (
        df.groupby(["risk_group", "tenure_segment"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
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
            avg_completion_rate=("completion_rate", "mean"),
            avg_recommendation_click_rate=("recommendation_click_rate", "mean"),
        )
        .reset_index()
    )
    tenure_subscription = tenure_subscription[tenure_subscription["customers"] >= 20].sort_values(
        ["customers", "churn_rate"], ascending=[False, False]
    )

    plan_device = (
        high.groupby(["subscription_type", "primary_device"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_last_login_days=("days_since_last_login", "mean"),
            avg_weekly_watch_minutes=("avg_watch_time_minutes_per_week", "mean"),
        )
        .reset_index()
    )
    plan_device = plan_device[plan_device["customers"] >= 15].sort_values(["customers", "churn_rate"], ascending=[False, False])

    profile_cols = [
        "days_since_last_login",
        "avg_watch_time_minutes_per_week",
        "watch_sessions_per_week",
        "completion_rate",
        "recommendation_click_rate",
        "app_rating",
    ]
    profile_rows = []
    for group_name, source in [
        ("All test", df),
        ("Top 10%", high),
        ("Focus target", focus),
    ]:
        row = {
            "group": group_name,
            "customers": len(source),
            "churn_rate": source["actual_churn"].mean(),
            "basic_share": (source["subscription_type"] == "Basic").mean(),
            "mobile_share": (source["primary_device"] == "Mobile").mean(),
        }
        row.update(source[profile_cols].mean().to_dict())
        profile_rows.append(row)
    feature_profile = pd.DataFrame(profile_rows)

    threshold = pd.read_csv(THRESHOLD_RESULT_PATH)
    model_result = pd.read_csv(MODEL_RESULT_PATH)
    best_model = model_result.sort_values("pr_auc", ascending=False).iloc[0]
    selected_threshold = threshold.sort_values("f1", ascending=False).iloc[0]

    topk_display = topk.assign(
        target_pct=topk["target_pct"].map(lambda v: f"Top {int(v * 100)}%"),
        churn_rate=topk["churn_rate"].map(pct),
        lift=topk["lift"].map(lambda v: num(v, 2) + "x"),
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

    high_tenure_display = high_tenure.assign(
        churn_rate=high_tenure["churn_rate"].map(pct),
        avg_probability=high_tenure["avg_probability"].map(pct),
        avg_last_login_days=high_tenure["avg_last_login_days"].map(lambda v: num(v, 1)),
        avg_weekly_watch_minutes=high_tenure["avg_weekly_watch_minutes"].map(lambda v: num(v, 1)),
        basic_share=high_tenure["basic_share"].map(pct),
        mobile_share=high_tenure["mobile_share"].map(pct),
    ).rename(
        columns={
            "tenure_segment": "Tenure segment",
            "customers": "Customers",
            "churn_rate": "Actual churn rate",
            "avg_probability": "Avg risk score",
            "avg_last_login_days": "Avg days since login",
            "avg_weekly_watch_minutes": "Avg weekly watch min",
            "basic_share": "Basic share",
            "mobile_share": "Mobile share",
        }
    )

    tenure_subscription_display = tenure_subscription.head(12).assign(
        churn_rate=tenure_subscription["churn_rate"].map(pct),
        avg_probability=tenure_subscription["avg_probability"].map(pct),
        avg_last_login_days=tenure_subscription["avg_last_login_days"].map(lambda v: num(v, 1)),
        avg_weekly_watch_minutes=tenure_subscription["avg_weekly_watch_minutes"].map(lambda v: num(v, 1)),
        avg_completion_rate=tenure_subscription["avg_completion_rate"].map(lambda v: num(v, 1)),
        avg_recommendation_click_rate=tenure_subscription["avg_recommendation_click_rate"].map(lambda v: num(v, 1)),
    ).rename(
        columns={
            "tenure_segment": "Tenure segment",
            "subscription_type": "Plan",
            "customers": "Customers",
            "churn_rate": "Actual churn rate",
            "avg_probability": "Avg risk score",
            "avg_last_login_days": "Avg days since login",
            "avg_weekly_watch_minutes": "Avg weekly watch min",
            "avg_completion_rate": "Avg completion",
            "avg_recommendation_click_rate": "Avg rec click",
        }
    )

    plan_device_display = plan_device.head(10).assign(
        churn_rate=plan_device["churn_rate"].map(pct),
        avg_last_login_days=plan_device["avg_last_login_days"].map(lambda v: num(v, 1)),
        avg_weekly_watch_minutes=plan_device["avg_weekly_watch_minutes"].map(lambda v: num(v, 1)),
    ).rename(
        columns={
            "subscription_type": "Plan",
            "primary_device": "Device",
            "customers": "Customers",
            "churn_rate": "Actual churn rate",
            "avg_last_login_days": "Avg days since login",
            "avg_weekly_watch_minutes": "Avg weekly watch min",
        }
    )

    feature_profile_display = feature_profile.assign(
        churn_rate=feature_profile["churn_rate"].map(pct),
        basic_share=feature_profile["basic_share"].map(pct),
        mobile_share=feature_profile["mobile_share"].map(pct),
        days_since_last_login=feature_profile["days_since_last_login"].map(lambda v: num(v, 1)),
        avg_watch_time_minutes_per_week=feature_profile["avg_watch_time_minutes_per_week"].map(lambda v: num(v, 1)),
        watch_sessions_per_week=feature_profile["watch_sessions_per_week"].map(lambda v: num(v, 1)),
        completion_rate=feature_profile["completion_rate"].map(lambda v: num(v, 1)),
        recommendation_click_rate=feature_profile["recommendation_click_rate"].map(lambda v: num(v, 1)),
        app_rating=feature_profile["app_rating"].map(lambda v: num(v, 2)),
    ).rename(
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
    )

    focus_churners_share = focus["actual_churn"].sum() / high["actual_churn"].sum()

    topk_chart = make_topk_chart(topk)
    tenure_chart = make_tenure_chart(high_tenure)
    heatmap = make_risk_tenure_heatmap(risk_tenure)
    activity_chart = make_activity_chart(high_activity)
    focus_profile_chart = make_focus_profile_chart(feature_profile)

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Top Risk Retention Strategy</title>
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
    body {{
      margin: 0;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
      background: #ffffff;
    }}
    header {{
      padding: 48px 40px 28px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    nav {{
      max-width: 1100px;
      margin: 0 auto 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    nav a {{
      color: var(--slate);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      background: #ffffff;
      font-size: 14px;
    }}
    nav a.active {{
      color: #ffffff;
      background: var(--blue);
      border-color: var(--blue);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 28px 64px;
    }}
    h1 {{
      max-width: 1100px;
      margin: 0 auto 12px;
      font-size: 34px;
      line-height: 1.18;
      letter-spacing: 0;
    }}
    .subtitle {{
      max-width: 1100px;
      margin: 0 auto;
      color: var(--muted);
      font-size: 16px;
    }}
    h2 {{
      margin: 36px 0 14px;
      padding-top: 8px;
      font-size: 23px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 24px 0 8px;
      font-size: 17px;
    }}
    p {{ margin: 8px 0 12px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0 28px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 14px 12px;
      background: #ffffff;
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric .value {{
      display: block;
      margin-top: 5px;
      font-size: 24px;
      font-weight: 700;
    }}
    .note {{
      border-left: 4px solid var(--blue);
      padding: 12px 14px;
      background: #eff6ff;
      color: #1e3a8a;
      margin: 16px 0 24px;
    }}
    .decision {{
      border: 1px solid #fecaca;
      border-left: 5px solid var(--red);
      border-radius: 8px;
      padding: 16px;
      background: #fff7f7;
      margin: 16px 0 24px;
    }}
    .decision strong {{
      display: block;
      margin-bottom: 6px;
      color: #991b1b;
      font-size: 17px;
    }}
    .chart {{
      margin: 14px 0 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #ffffff;
    }}
    .chart img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 22px;
      font-size: 14px;
    }}
    .data-table th {{
      text-align: left;
      border-bottom: 2px solid var(--slate);
      padding: 9px 8px;
      white-space: nowrap;
    }}
    .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    .strategy {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: var(--soft);
    }}
    .strategy strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--slate);
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
    }}
    li {{ margin: 4px 0; }}
    footer {{
      margin-top: 42px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
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
      <a href="modeling_methodology.html">Modeling Methodology</a>
      <a class="active" href="top_risk_retention_strategy.html">Retention Strategy</a>
    </nav>
    <h1>핵심 Top Risk 고객 집중 Retention Strategy</h1>
    <p class="subtitle">
      저장된 risk score 결과를 기반으로 이탈 위험 상위 고객을 다시 세분화하고,
      모든 구간에 무차별 액션을 배정하는 대신 규모와 이탈률이 모두 큰 핵심 타겟을 선정한 리포트입니다.
    </p>
  </header>

  <main>
    <section>
      <h2>1. 모델 결과 요약</h2>
      <div class="metrics">
        <div class="metric"><span class="label">Base churn rate</span><span class="value">{pct(base_churn_rate)}</span></div>
        <div class="metric"><span class="label">Final model</span><span class="value">{best_model["model"]}</span></div>
        <div class="metric"><span class="label">Best PR AUC</span><span class="value">{num(best_model["pr_auc"], 3)}</span></div>
        <div class="metric"><span class="label">Selected threshold F1</span><span class="value">{num(selected_threshold["f1"], 3)}</span></div>
      </div>
      <p>
        최종 모델은 모든 고객을 완벽히 0/1로 분류하기보다, 이탈 가능성이 높은 고객을 상위 구간에 모아
        캠페인 우선순위를 정하는 risk ranking 모델로 해석하는 것이 적절합니다.
      </p>
      {dataframe_to_html(topk_display)}
      <div class="chart"><img alt="Top-k targeting performance" src="data:image/png;base64,{topk_chart}" /></div>
    </section>

    <section>
      <h2>2. 추가 분석 질문</h2>
      <div class="note">
        기존 top-k 분석은 “상위 몇 %를 타겟팅할 것인가”에 답합니다.
        추가 분석은 “상위 위험군 안에서 어디에 먼저 예산을 쓸 것인가”에 답합니다.
      </div>
      <p>
        모든 세그먼트에 맞춤형 retention을 적용하면 실행 복잡도와 쿠폰 비용이 커집니다.
        따라서 상위 10% 고위험 고객 안에서 고객 수와 실제 이탈률을 함께 보고,
        가장 큰 비즈니스 효과가 기대되는 핵심 구간을 우선 타겟으로 선정합니다.
      </p>
    </section>

    <section>
      <h2>3. 상위 10% 고객의 가입기간별 규모</h2>
      {dataframe_to_html(high_tenure_display)}
      <div class="chart"><img alt="High risk tenure profile" src="data:image/png;base64,{tenure_chart}" /></div>
      <p>
        상위 10% 안에서는 모든 가입기간 세그먼트의 실제 이탈률이 매우 높습니다.
        다만 고객 수는 성장기와 장기 고객에 집중되어 있어, 캠페인 규모를 고려하면 이 두 집단을 먼저 검토하는 것이 합리적입니다.
      </p>
    </section>

    <section>
      <h2>4. 핵심 타겟 선정</h2>
      <div class="decision">
        <strong>선정 타겟: 상위 10% 중 성장기/장기 Basic 고객</strong>
        이 구간은 {len(focus):,}명으로 상위 10% 고객의 {pct(len(focus) / len(high))}를 차지하고,
        실제 churn rate는 {pct(focus["actual_churn"].mean())}입니다.
        또한 상위 10% 내 실제 이탈자의 {pct(focus_churners_share)}를 포함하므로,
        제한된 예산에서 가장 먼저 공략할 타겟으로 적합합니다.
      </div>
      <p>
        아래 표는 상위 10% 고객을 가입기간과 요금제로 나눈 결과입니다.
        성장기 Basic과 장기 Basic이 고객 수 기준 1, 2위이며, 두 구간 모두 87% 이상의 높은 이탈률을 보입니다.
      </p>
      {dataframe_to_html(tenure_subscription_display)}
    </section>

    <section>
      <h2>5. 추가 피처 기반 타겟 검증</h2>
      <p>
        핵심 타겟이 단순히 “Basic이라서” 선택된 것은 아닙니다.
        전체 고객 및 상위 10% 평균과 비교해도 최근 접속 공백, 낮은 시청시간, 낮은 완료율,
        낮은 추천 클릭률이 함께 나타납니다.
      </p>
      {dataframe_to_html(feature_profile_display)}
      <div class="chart"><img alt="Focus target feature profile" src="data:image/png;base64,{focus_profile_chart}" /></div>
    </section>

    <section>
      <h2>6. Risk Group x Tenure</h2>
      <div class="chart"><img alt="Risk group tenure heatmap" src="data:image/png;base64,{heatmap}" /></div>
      <p>
        Low risk 고객은 가입기간과 관계없이 churn rate가 낮지만, High risk 고객은 가입기간과 무관하게 높은 churn rate를 보입니다.
        따라서 1차 우선순위는 risk score로 정하고, 2차로 규모가 큰 Basic 고객군에 액션을 집중하는 방식이 적합합니다.
      </p>
    </section>

    <section>
      <h2>7. 가입기간 x 최근 접속 상태</h2>
      <div class="chart"><img alt="Tenure by activity segment" src="data:image/png;base64,{activity_chart}" /></div>
      <p>
        상위 10% 고객의 상당수는 31일 이상 미접속 고객입니다. 단순히 “이탈 위험이 높다”에서 끝내기보다,
        핵심 타겟 안에서도 장기 미접속 고객을 우선순위로 두면 캠페인 실행 대상을 더 좁힐 수 있습니다.
      </p>
    </section>

    <section>
      <h2>8. 요금제 x 기기 보조 인사이트</h2>
      <p>
        요금제와 기기를 함께 보면 Basic + Mobile 고객이 가장 큰 구간입니다.
        따라서 핵심 타겟 캠페인은 가격 민감도뿐 아니라 모바일 사용 맥락까지 반영하는 편이 좋습니다.
      </p>
      {dataframe_to_html(plan_device_display)}
    </section>

    <section>
      <h2>9. 핵심 타겟 전용 Retention Action</h2>
      <p>
        이번 리포트의 제안은 모든 고위험 구간에 서로 다른 캠페인을 배정하는 것이 아니라,
        규모가 가장 큰 성장기/장기 Basic 고객에게 먼저 예산을 집중하는 것입니다.
      </p>
      <div class="two-col">
        <div class="strategy">
          <strong>1순위: Basic 유지 혜택</strong>
          <p>가격 민감 고객의 즉시 해지를 방어하는 액션입니다.</p>
          <ul>
            <li>한시적 유지 할인</li>
            <li>Basic Plus 체험 제안</li>
            <li>Standard 업그레이드 1개월 무료 체험</li>
          </ul>
        </div>
        <div class="strategy">
          <strong>2순위: 모바일 재활성화</strong>
          <p>Basic 고객 중 Mobile 사용 비중이 높기 때문에 앱 복귀를 유도합니다.</p>
          <ul>
            <li>모바일 중심 고객에게 짧은 콘텐츠 큐레이션</li>
            <li>최근 인기 콘텐츠 푸시</li>
            <li>이어보기 기반 복귀 알림</li>
          </ul>
        </div>
        <div class="strategy">
          <strong>3순위: 추천 반응 회복</strong>
          <p>추천 클릭률과 완료율이 낮아 콘텐츠 발견 실패 가능성이 있습니다.</p>
          <ul>
            <li>최근 선호 장르 재탐색</li>
            <li>신작/인기작 중심 추천</li>
            <li>완주율 높은 짧은 콘텐츠 추천</li>
          </ul>
        </div>
        <div class="strategy">
          <strong>후순위: 소규모 구간 A/B 테스트</strong>
          <p>신규/초장기 구간은 위험도는 높지만 규모가 작아 별도 검증 대상으로 둡니다.</p>
          <ul>
            <li>신규 고객 온보딩 개선 실험</li>
            <li>초장기 고객 win-back 테스트</li>
            <li>표본 확장 후 별도 캠페인 판단</li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>10. 발표용 결론</h2>
      <p>
        모델은 이탈 위험 상위 10% 고객의 실제 churn rate를 {pct(topk.loc[topk["target_pct"] == 0.10, "churn_rate"].iloc[0])}까지
        끌어올려 캠페인 우선순위화에 유효했습니다. 추가 분석에서는 상위 위험군 전체에 무차별적으로 액션을 배정하지 않고,
        규모와 이탈률을 함께 고려해 성장기/장기 Basic 고객을 핵심 타겟으로 선정했습니다.
        이 구간은 상위 10% 고객의 {pct(len(focus) / len(high))}를 차지하고 실제 이탈률도 {pct(focus["actual_churn"].mean())}로 높아,
        Basic 유지 혜택과 모바일 재활성화 캠페인을 우선 적용하는 전략이 가장 실무적입니다.
      </p>
    </section>

    <footer>
      Source files: outputs/churn_risk_scores_test.csv, outputs/churn_model_final_result.csv, outputs/churn_threshold_result.csv
    </footer>
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
