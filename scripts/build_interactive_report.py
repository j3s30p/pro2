from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RISK_SCORE_PATH = ROOT / "outputs" / "churn_risk_scores_test.csv"
REPORT_PATH = ROOT / "reports" / "interactive_report.html"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


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
    return df


def build_payload(df: pd.DataFrame) -> dict[str, object]:
    total_churners = int(df["actual_churn"].sum())
    base_rate = df["actual_churn"].mean()
    topk = []
    for p in range(5, 31):
        top = df.head(int(len(df) * p / 100))
        topk.append(
            {
                "pct": p,
                "customers": len(top),
                "churnRate": round(float(top["actual_churn"].mean() * 100), 1),
                "captured": round(float(top["actual_churn"].sum() / total_churners * 100), 1),
                "lift": round(float(top["actual_churn"].mean() / base_rate), 2),
            }
        )

    high = df[df["risk_group"] == "High: top 10%"].copy()
    focus = high[
        high["tenure_segment"].isin(["Growing 13-36m", "Long 37-72m"])
        & (high["subscription_type"] == "Basic")
    ].copy()

    profiles = []
    feature_cols = [
        ("days_since_last_login", "Login gap"),
        ("avg_watch_time_minutes_per_week", "Watch min"),
        ("completion_rate", "Completion"),
        ("recommendation_click_rate", "Rec click"),
        ("app_rating", "App rating"),
    ]
    for name, source in [("All test", df), ("Top 10%", high), ("Focus target", focus)]:
        profiles.append(
            {
                "name": name,
                "customers": int(len(source)),
                "churnRate": round(float(source["actual_churn"].mean() * 100), 1),
                "metrics": [
                    {"label": label, "value": round(float(source[col].mean()), 1)}
                    for col, label in feature_cols
                ],
            }
        )

    tenure_plan = (
        high.groupby(["tenure_segment", "subscription_type"], observed=True)
        .agg(
            customers=("user_id", "size"),
            churn_rate=("actual_churn", "mean"),
            avg_probability=("churn_probability", "mean"),
        )
        .reset_index()
        .sort_values(["customers", "churn_rate"], ascending=[False, False])
        .head(8)
    )
    segment_rows = [
        {
            "segment": f"{row.tenure_segment} / {row.subscription_type}",
            "customers": int(row.customers),
            "churnRate": round(float(row.churn_rate * 100), 1),
            "risk": round(float(row.avg_probability * 100), 1),
        }
        for row in tenure_plan.itertuples(index=False)
    ]

    cluster = (
        df.groupby(["cluster", "risk_group"], observed=True)
        .agg(customers=("user_id", "size"), churn_rate=("actual_churn", "mean"))
        .reset_index()
    )
    cluster_rows = [
        {
            "cluster": int(row.cluster),
            "riskGroup": str(row.risk_group),
            "customers": int(row.customers),
            "churnRate": round(float(row.churn_rate * 100), 1),
        }
        for row in cluster.itertuples(index=False)
    ]

    return {
        "model": "Stacking (original features)",
        "totalCustomers": int(len(df)),
        "baseChurn": round(float(base_rate * 100), 1),
        "totalChurners": total_churners,
        "topk": topk,
        "profiles": profiles,
        "segments": segment_rows,
        "clusters": cluster_rows,
        "focus": {
            "customers": int(len(focus)),
            "share": round(float(len(focus) / len(high) * 100), 1),
            "churnRate": round(float(focus["actual_churn"].mean() * 100), 1),
        },
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = add_segments(pd.read_csv(RISK_SCORE_PATH).sort_values("churn_probability", ascending=False).reset_index(drop=True))
    payload = build_payload(df)
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive Churn Report</title>
  <style>
    :root {{
      --ink:#111827; --muted:#64748b; --line:#d9e2ec; --panel:#ffffff; --soft:#f5f7fb;
      --blue:#1f5fbf; --green:#287a47; --red:#b42318; --amber:#a65f00; --nav:#0f172a;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f8fafc; line-height:1.55; }}
    .shell {{ min-height:100vh; display:grid; grid-template-columns:280px minmax(0,1fr); }}
    aside {{ position:sticky; top:0; height:100vh; background:var(--nav); color:#e5e7eb; padding:26px 20px; overflow:auto; }}
    aside h1 {{ margin:0 0 8px; font-size:24px; line-height:1.18; letter-spacing:0; }}
    aside p {{ margin:0 0 22px; color:#aeb8c7; font-size:14px; }}
    .nav-btn {{ width:100%; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04); color:#e5e7eb; border-radius:8px; padding:11px 12px; margin:6px 0; text-align:left; cursor:pointer; font-weight:650; }}
    .nav-btn.active {{ background:#ffffff; color:var(--nav); border-color:#ffffff; }}
    .links {{ display:flex; flex-direction:column; gap:8px; margin-top:22px; }}
    .links a {{ color:#cbd5e1; text-decoration:none; font-size:13px; }}
    main {{ padding:28px; max-width:1180px; width:100%; }}
    .hero {{ background:#ffffff; border:1px solid var(--line); border-radius:10px; padding:26px; display:grid; grid-template-columns:1.3fr .9fr; gap:20px; align-items:end; }}
    .eyebrow {{ display:block; color:var(--muted); font-size:13px; font-weight:750; text-transform:uppercase; }}
    h2 {{ margin:6px 0 10px; font-size:31px; line-height:1.15; letter-spacing:0; }}
    h3 {{ margin:0 0 10px; font-size:21px; letter-spacing:0; }}
    p {{ margin:8px 0 12px; color:var(--muted); }}
    .kpis {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .kpi, .panel, .step, .strategy-card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .kpi {{ padding:14px; }}
    .kpi span {{ color:var(--muted); font-size:13px; font-weight:700; }}
    .kpi b {{ display:block; margin-top:4px; font-size:24px; line-height:1.15; overflow-wrap:anywhere; }}
    .stage {{ display:none; margin-top:20px; }}
    .stage.active {{ display:block; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .panel {{ padding:18px; margin-bottom:16px; }}
    .step {{ padding:16px; }}
    .step strong {{ display:block; margin:4px 0 6px; font-size:18px; }}
    .flow {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .flow .step:nth-child(1) {{ border-top:4px solid var(--blue); }}
    .flow .step:nth-child(2) {{ border-top:4px solid var(--amber); }}
    .flow .step:nth-child(3) {{ border-top:4px solid var(--green); }}
    .flow .step:nth-child(4) {{ border-top:4px solid var(--red); }}
    input[type=range] {{ width:100%; accent-color:var(--blue); }}
    .bar-list {{ display:grid; gap:10px; }}
    .bar-row {{ display:grid; grid-template-columns:150px minmax(0,1fr) 70px; gap:10px; align-items:center; font-size:14px; }}
    .bar-track {{ height:13px; background:#e8eef5; border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; background:var(--blue); border-radius:999px; width:0; transition:width .25s ease; }}
    .table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .table th {{ text-align:left; border-bottom:2px solid #334155; padding:9px 8px; background:#fbfcfe; white-space:nowrap; }}
    .table td {{ border-bottom:1px solid var(--line); padding:9px 8px; }}
    .table tr.highlight td {{ background:#fff7ed; font-weight:700; }}
    .profile-tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 14px; }}
    .chip {{ border:1px solid var(--line); border-radius:999px; background:#fff; padding:8px 12px; cursor:pointer; font-weight:650; color:#334155; }}
    .chip.active {{ background:var(--blue); color:#fff; border-color:var(--blue); }}
    .strategy-card {{ padding:18px; }}
    .strategy-card strong {{ display:block; font-size:19px; margin:3px 0 8px; }}
    .note {{ border-left:4px solid var(--blue); background:#eef6ff; border-radius:8px; padding:13px 15px; color:#174ea6; }}
    @media (max-width:900px) {{
      .shell {{ grid-template-columns:1fr; }}
      aside {{ position:relative; height:auto; }}
      main {{ padding:18px; }}
      .hero, .grid-2, .grid-3, .flow, .kpis {{ grid-template-columns:1fr; }}
      h2 {{ font-size:26px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Interactive Churn Report</h1>
      <p>가정 → 실험 → 결과 → 최종 흐름으로 보는 Netflix churn analysis</p>
      <button class="nav-btn active" data-stage="assumption">1. 가정</button>
      <button class="nav-btn" data-stage="experiment">2. 실험</button>
      <button class="nav-btn" data-stage="result">3. 결과</button>
      <button class="nav-btn" data-stage="final">4. 최종</button>
      <div class="links">
        <a href="index.html">Report Home</a>
        <a href="eda_analysis.html">EDA Evidence</a>
        <a href="modeling_methodology.html">Modeling Methodology</a>
        <a href="retention_analysis.html">Retention Analysis</a>
        <a href="top_risk_retention_strategy.html">Final Strategy</a>
      </div>
    </aside>
    <main>
      <section class="hero">
        <div>
          <span class="eyebrow">Final model</span>
          <h2>Stacking (original features) 기반 risk ranking</h2>
          <p>노트북의 진행 구조를 유지하면서 EDA, 모델링, top-k 분석, 최종 retention action을 한 페이지에서 조작해볼 수 있도록 구성했습니다.</p>
        </div>
        <div class="kpis">
          <div class="kpi"><span>Customers</span><b id="kpi-customers"></b></div>
          <div class="kpi"><span>Base churn</span><b id="kpi-base"></b></div>
          <div class="kpi"><span>Focus target</span><b id="kpi-focus"></b></div>
        </div>
      </section>

      <section id="assumption" class="stage active">
        <div class="panel">
          <h3>가정</h3>
          <div class="flow">
            <div class="step"><span class="eyebrow">01 EDA</span><strong>churn 신호가 존재한다</strong><p>최근 로그인 공백, 시청량, 완료율, 추천 클릭률이 churn과 함께 움직이는지 확인합니다.</p></div>
            <div class="step"><span class="eyebrow">04-07 Modeling</span><strong>모델로 검증한다</strong><p>원본 feature와 engineered feature를 비교해 risk score 품질을 검증합니다.</p></div>
            <div class="step"><span class="eyebrow">06 Retention</span><strong>ranking으로 운영한다</strong><p>threshold가 아니라 상위 k% 고객을 잘 포착하는지 확인합니다.</p></div>
            <div class="step"><span class="eyebrow">Final Strategy</span><strong>핵심 타겟만 실행한다</strong><p>상위 10% 안에서 규모와 이탈률이 큰 구간에 비용이 드는 액션을 집중합니다.</p></div>
          </div>
        </div>
      </section>

      <section id="experiment" class="stage">
        <div class="grid-2">
          <div class="panel">
            <h3>Top-k 실험</h3>
            <p>슬라이더를 움직이면 risk score 상위 k%의 실제 churn rate, 전체 이탈자 포착률, lift가 바뀝니다.</p>
            <input id="topk-slider" type="range" min="5" max="30" value="10" />
            <div class="kpis">
              <div class="kpi"><span>Top-k</span><b id="topk-label"></b></div>
              <div class="kpi"><span>Churn rate</span><b id="topk-churn"></b></div>
              <div class="kpi"><span>Captured</span><b id="topk-captured"></b></div>
            </div>
            <div class="note" id="topk-note"></div>
          </div>
          <div class="panel">
            <h3>모델링 실험 요약</h3>
            <table class="table">
              <tbody>
                <tr><th>Baseline</th><td>Logistic Regression이 강한 기준선을 형성</td></tr>
                <tr><th>Ensemble</th><td>Stacking이 PR AUC 최고이나 개선 폭은 작음</td></tr>
                <tr><th>Feature Engineering</th><td>원본 feature 대비 일관된 개선 없음</td></tr>
                <tr><th>Decision</th><td>Stacking (original features) 유지</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="result" class="stage">
        <div class="grid-2">
          <div class="panel">
            <h3>고위험 고객 프로파일</h3>
            <div class="profile-tabs" id="profile-tabs"></div>
            <div class="bar-list" id="profile-bars"></div>
          </div>
          <div class="panel">
            <h3>Top 10% 내부 세그먼트</h3>
            <table class="table" id="segment-table"></table>
          </div>
        </div>
        <div class="panel">
          <h3>KMeans × Risk group</h3>
          <div class="bar-list" id="cluster-bars"></div>
        </div>
      </section>

      <section id="final" class="stage">
        <div class="grid-3">
          <div class="strategy-card"><span class="eyebrow">Target</span><strong>성장기/장기 Basic 고객</strong><p>Top 10% risk 고객 중 586명, focus target churn rate 88.6%.</p></div>
          <div class="strategy-card"><span class="eyebrow">Action 1</span><strong>업그레이드 쿠폰</strong><p>중/장기 Basic 고위험 고객에게 Standard 1개월 체험을 제공합니다.</p></div>
          <div class="strategy-card"><span class="eyebrow">Action 2</span><strong>Mobile-only 요금제</strong><p>모바일 사용 맥락과 가격 부담 proxy를 반영해 저가 유지 선택지를 제공합니다.</p></div>
        </div>
        <div class="panel">
          <h3>운영 원칙</h3>
          <table class="table">
            <tbody>
              <tr><th>우선순위</th><td>최종 모델의 churn_probability ranking 기준</td></tr>
              <tr><th>집행 범위</th><td>상위 10% pilot 이후 효과 검증</td></tr>
              <tr><th>성공 지표</th><td>30일 잔존율, 60일 잔존율, 업그레이드 유지율, ARPU guardrail</td></tr>
              <tr><th>주의</th><td>가격 민감도는 직접 관측값이 아니라 proxy 기반 보조 해석</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const fmt = (v) => `${{v}}%`;
    document.getElementById('kpi-customers').textContent = data.totalCustomers.toLocaleString();
    document.getElementById('kpi-base').textContent = fmt(data.baseChurn);
    document.getElementById('kpi-focus').textContent = data.focus.customers.toLocaleString();

    document.querySelectorAll('.nav-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.stage).classList.add('active');
      }});
    }});

    const slider = document.getElementById('topk-slider');
    function renderTopk() {{
      const row = data.topk.find(d => d.pct === Number(slider.value));
      document.getElementById('topk-label').textContent = `Top ${{row.pct}}%`;
      document.getElementById('topk-churn').textContent = fmt(row.churnRate);
      document.getElementById('topk-captured').textContent = fmt(row.captured);
      document.getElementById('topk-note').textContent = `${{row.customers.toLocaleString()}}명을 대상으로 할 때 lift는 ${{row.lift}}x입니다. Top 30%는 검증 범위, Top 10%는 초기 실행 범위로 해석합니다.`;
    }}
    slider.addEventListener('input', renderTopk);
    renderTopk();

    const tabs = document.getElementById('profile-tabs');
    const bars = document.getElementById('profile-bars');
    function renderProfile(name) {{
      const profile = data.profiles.find(p => p.name === name);
      document.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c.dataset.name === name));
      const max = Math.max(...profile.metrics.map(m => m.value), 1);
      bars.innerHTML = profile.metrics.map(m => `
        <div class="bar-row">
          <span>${{m.label}}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(4, m.value / max * 100)}}%"></div></div>
          <b>${{m.value}}</b>
        </div>
      `).join('');
    }}
    tabs.innerHTML = data.profiles.map((p, i) => `<button class="chip ${{i === 0 ? 'active' : ''}}" data-name="${{p.name}}">${{p.name}}</button>`).join('');
    tabs.querySelectorAll('.chip').forEach(chip => chip.addEventListener('click', () => renderProfile(chip.dataset.name)));
    renderProfile(data.profiles[0].name);

    const segmentTable = document.getElementById('segment-table');
    segmentTable.innerHTML = `<thead><tr><th>Segment</th><th>Customers</th><th>Churn</th><th>Avg risk</th></tr></thead><tbody>${
      data.segments.map(r => `<tr class="${{r.segment.includes('Basic') ? 'highlight' : ''}}"><td>${{r.segment}}</td><td>${{r.customers}}</td><td>${{r.churnRate}}%</td><td>${{r.risk}}%</td></tr>`).join('')
    }</tbody>`;

    const clusterBars = document.getElementById('cluster-bars');
    const highClusters = data.clusters.filter(c => c.riskGroup === 'High: top 10%');
    const maxCluster = Math.max(...highClusters.map(c => c.customers), 1);
    clusterBars.innerHTML = highClusters.map(c => `
      <div class="bar-row">
        <span>Cluster ${{c.cluster}}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(4, c.customers / maxCluster * 100)}}%; background:var(--green);"></div></div>
        <b>${{c.churnRate}}%</b>
      </div>
    `).join('');
  </script>
</body>
</html>
"""
    html = html.replace("{{", "{").replace("}}", "}").replace("__PAYLOAD_JSON__", payload_json)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
