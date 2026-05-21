# Netflix User Churn Analysis

Netflix-style user behavior data를 활용해 고객 세그먼트, 군집화, churn 예측 모델링, 그리고 이탈 방지 캠페인 액션 전략까지 분석한 프로젝트입니다.

## Project Goal

이 프로젝트의 목적은 단순히 churn 여부를 0/1로 맞추는 것이 아니라, 이탈 가능성이 높은 고객을 우선순위화해 retention campaign에 활용할 수 있는 분석 흐름을 만드는 것입니다.

주요 분석 질문은 다음과 같습니다.

- 어떤 고객 행동 특성이 churn과 관련이 있는가?
- 고객을 의미 있는 세그먼트로 나눌 수 있는가?
- churn 예측 모델은 어느 정도의 성능을 보이는가?
- 모델의 예측 확률을 활용해 이탈 위험 상위 고객군을 식별할 수 있는가?
- 상위 위험군에 어떤 retention action을 제안할 수 있는가?

## Repository Structure

```text
.
├── data/
│   ├── KKBOX/
│   │   └── kkbox_netflix_format.csv
│   └── user_behavior_50000/
│       ├── netflix_user_behavior_churn_50000.csv
│       └── netflix_user_behavior_churn_50000v2.csv
├── notebooks/
│   ├── 01_eda_segment_insight.ipynb
│   ├── 02_clustering_binned.ipynb
│   ├── 03_clustering_scaled.ipynb
│   ├── 04_churn_modeling_baseline.ipynb
│   ├── 05_churn_modeling_tuning_ensemble.ipynb
│   ├── 06_churn_model_interpretation_action.ipynb
│   ├── 07-1_churn_feature_engineering_first_try.ipynb
│   ├── 07-2_churn_behavior_feature_engineering_pipeline.ipynb
│   ├── 07-3_churn_auc_f1_optimization.ipynb
│   ├── 07-4_churn_f1_optimization_light.ipynb
│   ├── 07-5_churn_paper_based_feature_engineering.ipynb
│   └── 08_kkbox_transfer_indomain.ipynb
├── outputs/
│   ├── churn_risk_scores_test.csv
│   ├── churn_model_final_result.csv
│   ├── churn_threshold_result.csv
│   ├── final_churn_model_eval.csv
│   ├── final_churn_risk_scores_test.csv
│   └── final_churn_topk_metrics.csv
├── models/
│   ├── final_churn_stacking_pipeline.joblib
│   └── final_churn_model_metadata.json
├── report/
│   ├── index.html
│   ├── eda.html
│   ├── clustering.html
│   ├── feature_engineering.html
│   ├── modeling.html
│   ├── results.html
│   ├── business_insight.html
│   └── style.css
├── pyproject.toml
├── uv.lock
└── README.md
```

`outputs/` 폴더는 `05`와 `07` 계열 노트북의 저장 섹션을 실행하면 생성됩니다. `06` 노트북은 저장된 risk score와 성능 결과를 읽어 분석합니다.

## Environment

이 프로젝트는 `uv` 기반으로 관리합니다.

```bash
uv sync
```

노트북 실행:

```bash
uv run jupyter lab
```

현재 프로젝트 환경은 `.python-version`과 `pyproject.toml` 기준 Python 3.14를 사용합니다.

## Notebook Flow

권장 실행 순서는 다음과 같습니다.

1. `01_eda_segment_insight.ipynb`
   - 데이터 구조 확인
   - EDA
   - 세그먼트별 churn insight 도출

2. `02_clustering_binned.ipynb`
   - 수치형 변수를 범주화한 뒤 KMeans 군집화
   - 군집별 churn rate와 고객 특성 해석

3. `03_clustering_scaled.ipynb`
   - 수치형 scaling + 범주형 one-hot 기반 KMeans 군집화
   - 최종 cluster profile 해석

4. `04_churn_modeling_baseline.ipynb`
   - baseline churn prediction
   - Dummy, Logistic Regression, Random Forest, boosting 후보 비교
   - threshold 조정
   - feature importance
   - cluster label 추가 모델 비교

5. `05_churn_modeling_tuning_ensemble.ipynb`
   - Stratified K-Fold CV
   - GridSearchCV / RandomizedSearchCV
   - Soft Voting
   - Stacking
   - 최종 risk score 저장

6. `06_churn_model_interpretation_action.ipynb`
   - 저장된 risk score 로드
   - decile analysis
   - lift chart
   - top-k campaign targeting
   - risk group profiling
   - cluster × risk 결합 분석
   - retention action matrix

7. `07-1_churn_feature_engineering_first_try.ipynb`
   - ratio, risk flag, interaction, bin 기반 1차 feature engineering
   - 원본 Stacking 모델과 engineered feature set 비교
   - top-k targeting 관점에서 개선 여부 확인

8. `07-2_churn_behavior_feature_engineering_pipeline.ipynb`
   - train 기준 quantile을 fit하는 leakage-safe behavior feature pipeline
   - `inactive_low_watch_flag`, `low_interest_flag`, `basic_mobile_flag`, `engagement_score` 등 생성
   - 최종 모델 입력으로 채택하지 않고, 성능 개선이 제한적이었다는 실험 근거와 행동 해석 보조 자료로 정리

9. `07-3_churn_auc_f1_optimization.ipynb`
   - ROC AUC와 F1 개선 목적의 compact feature 및 모델 탐색
   - Logistic Regression, LightGBM, XGBoost, CatBoost, soft average ensemble 비교
   - ranking 목적과 threshold 분류 목적의 후보가 다를 수 있음을 확인

10. `07-4_churn_f1_optimization_light.ipynb`
    - F1 기준 lightweight search
    - valid threshold tuning 후 test set에서 최종 확인
    - F1 직접 최적화가 기존 최고 후보를 넘는지 검증

11. `07-5_churn_paper_based_feature_engineering.ipynb`
    - OTT 구독 해지 관련 논문을 바탕으로 paper-based feature engineering 수행
    - 가격 대비 가치, 추천/개인화 반응, 전환 위험, 구독 피로감, 몰아보기/콘텐츠 고갈 proxy 생성
    - 성능 개선보다는 해석용 진단 피처로 가치가 있음을 확인

12. `08_kkbox_transfer_indomain.ipynb`
    - KKBOX 실제 구독 데이터에 동일 Stacking 파이프라인 적용
    - 합성 데이터 대비 실제 데이터에서 성능 대폭 향상 확인 (ROC AUC 0.977, F1 0.881)
    - 파이프라인의 실무 적용 가능성 검증

## HTML Report

분석 결과를 발표/공유하기 쉽게 HTML 리포트로 정리했습니다.
`report/` 디렉토리에 정적 HTML 페이지로 구성되어 있으며, GitHub Pages로 배포됩니다.

- `report/index.html` — 프로젝트 개요 및 섹션 네비게이션
- `report/eda.html` — 탐색적 데이터 분석 (타겟 분포, 상관관계, 세그먼트별 이탈률)
- `report/clustering.html` — KMeans 고객 군집화 (k=3, 고위험/일반/충성 군집)
- `report/feature_engineering.html` — 피처 엔지니어링 실험 5회 (고정 임계값, 데이터 기반, 논문 기반)
- `report/modeling.html` — 모델링 파이프라인 (Baseline → CV → Tuning → Ensemble → Threshold)
- `report/results.html` — 모델 예측 결과 (Decile, Top-K, Risk Group, KMeans 교차 검증)
- `report/business_insight.html` — 리텐션 전략, A/B 테스트 설계, KKBOX 검증

## Key Results

모델링 단계에서는 Logistic Regression이 강한 기준선을 만들었고, 튜닝 및 앙상블 이후 Stacking이 가장 높은 PR AUC를 기록했습니다. 이후 feature engineering 실험에서도 개선 폭이 제한적이어서 최종 저장 모델은 **Stacking (original features)** 로 정리했습니다.

따라서 최종 해석은 classification score보다 ranking 활용에 집중했습니다.

Top-k 비율 산출 방식:

- 최종 모델 `Stacking (original features)`가 산출한 `churn_probability`를 기준으로 test 고객 10,000명을 내림차순 정렬했습니다.
- 상위 10%, 20%, 30%는 이 정렬 결과에서 각각 상위 1,000명, 2,000명, 3,000명을 절단한 구간입니다.
- `Actual churn rate`는 해당 구간 내 실제 이탈자 수를 해당 구간 고객 수로 나눈 값입니다.
- `Captured churners`는 해당 구간 내 실제 이탈자 수를 전체 test 이탈자 2,093명으로 나눈 값입니다.
- `Lift`는 전체 test churn rate 대비 해당 구간 churn rate의 배율입니다.
- threshold는 0/1 분류 성능 비교용이며, top-k ranking 구간 생성에는 사용하지 않았습니다.

주요 결과:

- 최종 모델: Stacking (original features)
- 최종 모델 PR AUC: 약 0.764
- threshold 조정 후 f1: 약 0.694
- 이탈 확률 상위 10% 고객의 실제 churn rate: 약 87.6%
- 상위 10% lift: 약 4.19배
- 상위 20% 타겟팅 시 전체 이탈자의 약 67.9% 포착
- 상위 30% 타겟팅 시 전체 이탈자의 약 82.0% 포착

Feature engineering 실험 결과:

- 1차 feature engineering과 behavior feature engineering은 원본 feature set 대비 PR AUC와 top 10% lift를 일관되게 개선하지 못했습니다.
- 논문 기반 feature engineering도 test F1, ROC AUC, PR AUC 기준으로 원본 feature set을 넘지 못했습니다.
- 따라서 최종 risk score 산출은 원본 feature 기반 Stacking 모델을 유지했습니다.
- Engineered feature와 논문 기반 proxy feature는 최종 모델 입력으로 채택하지 않았으며, 성능 개선이 제한적이었다는 실험 근거와 고위험 고객 해석을 보조하는 참고 자료로 활용했습니다.

## Final Interpretation

이 모델은 모든 고객을 완벽히 churn / non-churn으로 분류하는 모델이라기보다, 이탈 방지 캠페인의 우선순위를 정하는 risk ranking 모델로 활용하는 것이 적절합니다.

추천 운영 방향:

- High risk: 즉시 개입 대상
- Medium risk: 저비용 캠페인 대상
- Low risk: 고비용 캠페인 제외 대상

고위험 고객은 최근 로그인 공백이 길고, 주간 시청 시간이 낮으며, completion rate와 recommendation click rate가 낮은 특징을 보였습니다. Basic 요금제와 Mobile 사용 비중도 높았습니다.

추가 HTML 리포트에서는 모든 고위험 구간에 무차별적으로 맞춤형 retention을 적용하기보다, 상위 10% 고객 안에서 규모와 이탈률이 모두 큰 **성장기/장기 Basic 고객**을 핵심 타겟으로 선정했습니다.

상위 30% risk 고객이 전체 이탈자의 약 82.0%를 포함한다는 점은 모델의 ranking 성능을 보여주는 근거로 해석했습니다. 다만 실제 캠페인은 쿠폰 비용, 메시지 피로도, 운영 복잡도가 발생하므로 상위 30% 전체를 바로 타겟팅하지 않고, 실제 churn rate가 약 87.6%로 가장 높은 상위 10%를 1차 실행 대상으로 설정했습니다.

또한 risk group은 모델 예측확률 기반 실행 우선순위이고, KMeans cluster는 `05` 노트북에서 train 데이터로만 직접 fit한 비지도 고객 군집입니다. 두 결과를 결합한 결과 High risk 고객 대부분이 특정 KMeans cluster에 집중되어, risk score 기반 타겟이 행동 패턴 측면에서도 일관된 고객군임을 확인했습니다.

핵심 타겟 요약:

- 상위 10% risk 고객 중 성장기/장기 Basic 고객: 586명
- 상위 10% 고객의 약 58.6%
- 실제 churn rate: 약 88.6%
- 상위 10% 내 실제 이탈자의 약 59.2% 포함

따라서 우선 액션은 두 가지로 정리할 수 있습니다. 첫째, Mobile 사용 비중이 높은 가격 민감 고객에게는 모바일에서만 사용할 수 있지만 더 저렴한 **Mobile-only 요금제**를 제안해 완전 해지를 저가 유지로 전환합니다. 둘째, 성장기/장기 Basic 고객 중 risk score가 높은 고객에게만 **Standard 업그레이드 1개월 쿠폰**을 제공해 상위 요금제 가치를 체험하게 합니다.

이 업그레이드 쿠폰은 신규 고객 확보용 무료 체험권이 아니라, 과거 Netflix가 일부 국가에서 신규 가입자 대상으로 운영했던 한시적 요금제 업그레이드 프로모션 방식을 기존 고위험 고객 retention 전략으로 재해석한 것입니다.

## Notes Before Running

- `06` 노트북은 모델을 다시 학습하지 않습니다.
- 먼저 `05` 노트북 마지막의 Risk Score 저장 섹션을 실행해야 합니다.
- `07` 계열 노트북은 추가 feature engineering / optimization 실험 기록입니다. 최종 운영 결론은 `Stacking (original features)` 기반 risk score를 유지하는 쪽입니다.
- `report/` HTML 파일은 직접 작성한 정적 리포트입니다.
- `catboost_info/`, `.venv/`, `.ipynb_checkpoints/`, `.DS_Store`, `.matplotlib-cache/` 등은 `.gitignore`로 제외합니다.
