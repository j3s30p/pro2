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
│   └── user_behavior_50000/
│       ├── netflix_user_behavior_churn_50000.csv
│       └── netflix_user_behavior_churn_50000v2.csv
├── notebooks/
│   ├── 01_eda_segment_insight.ipynb
│   ├── 02_clustering_binned.ipynb
│   ├── 03_clustering_scaled.ipynb
│   ├── 04_churn_modeling_baseline.ipynb
│   ├── 05_churn_modeling_tuning_ensemble.ipynb
│   └── 06_churn_model_interpretation_action.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

`outputs/` 폴더는 `05` 노트북의 Risk Score 저장 섹션을 실행하면 생성됩니다. `06` 노트북은 이 저장 결과를 읽어 분석합니다.

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

## Key Results

모델링 단계에서는 Logistic Regression이 강한 기준선을 만들었고, 튜닝 및 앙상블 이후 Stacking이 가장 높은 PR AUC를 기록했습니다. 다만 단일 모델 대비 개선 폭은 작았습니다.

따라서 최종 해석은 classification score보다 ranking 활용에 집중했습니다.

주요 결과:

- 최종 후보 모델: Stacking
- 최종 모델 PR AUC: 약 0.764
- threshold 조정 후 f1: 약 0.694
- 이탈 확률 상위 10% 고객의 실제 churn rate: 약 87.6%
- 상위 10% lift: 약 4.19배
- 상위 20% 타겟팅 시 전체 이탈자의 약 67.9% 포착
- 상위 30% 타겟팅 시 전체 이탈자의 약 82.0% 포착

## Final Interpretation

이 모델은 모든 고객을 완벽히 churn / non-churn으로 분류하는 모델이라기보다, 이탈 방지 캠페인의 우선순위를 정하는 risk ranking 모델로 활용하는 것이 적절합니다.

추천 운영 방향:

- High risk: 즉시 개입 대상
- Medium risk: 저비용 캠페인 대상
- Low risk: 고비용 캠페인 제외 대상

고위험 고객은 최근 로그인 공백이 길고, 주간 시청 시간이 낮으며, completion rate와 recommendation click rate가 낮은 특징을 보였습니다. Basic 요금제와 Mobile 사용 비중도 높아, 복귀 유도 메시지, 모바일 친화 추천, 짧은 콘텐츠 큐레이션, 요금제 혜택 제안이 적합합니다.

## Notes Before Running

- `06` 노트북은 모델을 다시 학습하지 않습니다.
- 먼저 `05` 노트북 마지막의 Risk Score 저장 섹션을 실행해야 합니다.
- `catboost_info/`, `.venv/`, `.ipynb_checkpoints/`, `.DS_Store` 등은 `.gitignore`로 제외합니다.
