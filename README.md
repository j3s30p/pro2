# Netflix User Churn Analysis

Netflix-style user behavior 데이터를 활용해 고객 이탈을 예측하고, 예측 확률을 기반으로 리텐션 캠페인 우선순위를 설계한 프로젝트입니다.

단순히 churn 여부를 0/1로 맞추는 것보다, 제한된 마케팅 예산 안에서 **누구를 먼저 잡아야 하는지**를 정하는 risk ranking 관점에 초점을 맞췄습니다.

정적 HTML 리포트는 GitHub Pages로 배포됩니다.

- Report: https://j3s30p.github.io/pro2/
- Repository: https://github.com/j3s30p/pro2

## Project Goal

이 프로젝트의 핵심 질문은 다음과 같습니다.

- 어떤 고객 행동 특성이 churn과 관련이 있는가?
- 고객을 행동 기반 군집으로 나눌 수 있는가?
- churn 예측 모델은 어느 정도의 성능을 보이는가?
- 예측 확률을 이용해 이탈 위험 상위 고객을 선별할 수 있는가?
- 고위험 고객에게 어떤 retention action을 제안할 수 있는가?
- 합성 데이터에서 만든 파이프라인이 실제 구독 데이터에서도 의미 있게 작동하는가?

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── pages.yml
├── data/
│   ├── KKBOX/
│   │   └── kkbox_netflix_format.csv
│   └── user_behavior_50000/
│       ├── netflix_user_behavior_churn_50000.csv
│       └── netflix_user_behavior_churn_50000v2.csv
├── models/
│   ├── final_churn_model_metadata.json
│   └── final_churn_stacking_pipeline.joblib
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
│   ├── churn_model_final_result.csv
│   ├── churn_risk_scores_test.csv
│   ├── churn_threshold_result.csv
│   ├── experiment_comparison.csv
│   ├── final_churn_model_eval.csv
│   ├── final_churn_risk_scores_test.csv
│   ├── final_churn_topk_metrics.csv
│   ├── kkbox_indomain_result.csv
│   └── kkbox_transfer_result.csv
├── report/
│   ├── images/
│   │   └── *.png
│   ├── index.html
│   ├── eda.html
│   ├── clustering.html
│   ├── feature_engineering.html
│   ├── modeling.html
│   ├── results.html
│   ├── business_insight.html
│   └── style.css
├── .gitattributes
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment

이 프로젝트는 `uv` 기반으로 관리합니다.

```bash
uv sync
uv run jupyter lab
```

환경 기준:

- Python: `>=3.14`
- 주요 라이브러리: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, `catboost`, `imblearn`
- KKBOX 데이터는 Git LFS로 관리됩니다. 처음 클론한 경우 `git lfs pull`이 필요할 수 있습니다.

## Data

주요 데이터는 두 종류입니다.

| Path | Description |
| --- | --- |
| `data/user_behavior_50000/netflix_user_behavior_churn_50000v2.csv` | 메인 분석에 사용한 Netflix-style 합성 고객 행동 데이터 |
| `data/user_behavior_50000/netflix_user_behavior_churn_50000.csv` | 이전 버전의 Netflix-style 데이터 |
| `data/KKBOX/kkbox_netflix_format.csv` | KKBOX 실제 구독 데이터를 프로젝트 피처 형식에 맞춘 검증용 데이터 |

Netflix-style 데이터는 합성 데이터이므로 모델 성능과 일반화 해석에 한계가 있습니다. 이 한계를 확인하기 위해 `08_kkbox_transfer_indomain.ipynb`에서 실제 구독 서비스 데이터인 KKBOX로 추가 검증을 진행했습니다.

## Notebook Flow

권장 실행 순서는 다음과 같습니다.

1. `01_eda_segment_insight.ipynb`
   - 데이터 구조 확인
   - 타겟 분포, 상관관계, 세그먼트별 churn rate 분석

2. `02_clustering_binned.ipynb`
   - 수치형 변수를 구간화한 뒤 KMeans 군집화
   - 군집별 churn rate와 고객 특성 해석

3. `03_clustering_scaled.ipynb`
   - 수치형 scaling + 범주형 one-hot 기반 KMeans 군집화
   - 최종 고객 군집 프로파일 정리

4. `04_churn_modeling_baseline.ipynb`
   - baseline churn prediction
   - Dummy, Logistic Regression, Random Forest, boosting 모델 비교
   - threshold 조정과 feature importance 확인

5. `05_churn_modeling_tuning_ensemble.ipynb`
   - Stratified K-Fold CV
   - GridSearchCV / RandomizedSearchCV
   - Soft Voting, Stacking 비교
   - 최종 모델과 risk score 저장

6. `06_churn_model_interpretation_action.ipynb`
   - 저장된 risk score 로드
   - decile analysis, lift chart, top-k targeting
   - risk group profiling
   - cluster x risk 결합 분석
   - retention action matrix 설계

7. `07-1_churn_feature_engineering_first_try.ipynb`
   - ratio, risk flag, interaction, bin 기반 1차 파생변수 실험
   - 원본 피처 대비 성능 개선 여부 확인

8. `07-2_churn_behavior_feature_engineering_pipeline.ipynb`
   - train 기준 임계값을 고정 적용하는 leakage-safe behavior feature pipeline 구성
   - 행동 해석용 파생변수의 실험 가치 확인

9. `07-3_churn_auc_f1_optimization.ipynb`
   - ROC AUC와 F1 개선 목적의 compact feature 및 모델 탐색
   - ranking 목적과 threshold 분류 목적의 후보가 다를 수 있음을 확인

10. `07-4_churn_f1_optimization_light.ipynb`
    - F1 기준 lightweight search
    - threshold tuning 후 test set 성능 확인

11. `07-5_churn_paper_based_feature_engineering.ipynb`
    - OTT 구독 해지 관련 논문을 바탕으로 proxy feature 생성
    - 가격 대비 가치, 추천 반응, 구독 피로감, 콘텐츠 고갈 신호 등을 실험

12. `08_kkbox_transfer_indomain.ipynb`
    - Netflix-style 데이터에서 만든 파이프라인을 KKBOX 데이터에 적용
    - synthetic 데이터 한계와 실제 데이터 적용 가능성을 비교 검증

## Outputs

`outputs/`에는 모델 평가와 리포트 작성에 사용한 중간/최종 결과가 저장되어 있습니다.

| File | Description |
| --- | --- |
| `churn_model_final_result.csv` | 모델별 최종 성능 비교 |
| `churn_risk_scores_test.csv` | 테스트셋 churn risk score |
| `churn_threshold_result.csv` | threshold별 Precision, Recall, F1 결과 |
| `experiment_comparison.csv` | feature engineering 실험 비교 |
| `final_churn_model_eval.csv` | 최종 모델 평가 결과 |
| `final_churn_risk_scores_test.csv` | 최종 risk score 결과 |
| `final_churn_topk_metrics.csv` | Top-K targeting 지표 |
| `kkbox_transfer_result.csv` | Netflix 학습 모델을 KKBOX에 전이 적용한 결과 |
| `kkbox_indomain_result.csv` | KKBOX 데이터 내 학습/평가 결과 |

`models/`에는 최종 Stacking pipeline과 메타데이터가 저장되어 있습니다.

## HTML Report

`report/`는 GitHub Pages에 배포되는 정적 리포트 소스입니다. HTML과 CSS를 직접 관리하며, 차트 이미지는 `report/images/`에 저장되어 있습니다.

| Page | Content |
| --- | --- |
| `report/index.html` | 프로젝트 개요와 전체 흐름 |
| `report/eda.html` | EDA, 타겟 분포, 상관관계, 세그먼트별 이탈률 |
| `report/clustering.html` | KMeans 고객 군집화와 군집별 churn profile |
| `report/feature_engineering.html` | 5회 feature engineering 실험과 미채택 사유 |
| `report/modeling.html` | Baseline, CV, tuning, ensemble, threshold 최적화 |
| `report/results.html` | Decile, Top-K, Lift, Risk Group, KMeans 교차 검증 |
| `report/business_insight.html` | 리텐션 전략, 캠페인 운영 원칙, KKBOX 검증, 최종 결론 |

GitHub Pages 배포는 `.github/workflows/pages.yml`에서 관리합니다. `main` 브랜치에 `report/**` 변경사항이 push되면 자동 배포됩니다.

## Key Results

최종 모델은 **Stacking (original features)** 입니다. 여러 feature engineering 실험을 진행했지만, 원본 피처 기반 모델이 가장 안정적인 PR AUC와 Top-K 성능을 보였습니다.

주요 성능:

| Metric | Result |
| --- | ---: |
| Final model | Stacking (original features) |
| PR AUC | 약 0.764 |
| F1 Score | 약 0.694 |
| Top 10% churn rate | 약 87.6% |
| Top 10% lift | 약 4.19x |
| Top 20% captured churners | 약 67.9% |
| Top 30% captured churners | 약 82.0% |

Top-K 산출 방식:

- 최종 모델의 `churn_probability`를 기준으로 test 고객 10,000명을 내림차순 정렬했습니다.
- 상위 10%, 20%, 30%는 각각 상위 1,000명, 2,000명, 3,000명을 의미합니다.
- `Actual churn rate`는 해당 구간 내 실제 이탈자 비율입니다.
- `Captured churners`는 해당 구간이 전체 이탈자 중 몇 퍼센트를 포함하는지 나타냅니다.
- `Lift`는 전체 test churn rate 대비 해당 구간 churn rate의 배율입니다.
- threshold는 0/1 분류 성능 비교용이며, Top-K ranking 구간 생성에는 사용하지 않았습니다.

## Feature Engineering Conclusion

Feature engineering은 총 5회 실험했습니다.

- 고정 임계값 기반 risk flag
- behavior feature pipeline
- compact feature 기반 AUC/F1 최적화
- lightweight F1 탐색
- 논문 기반 proxy feature

결론적으로, 파생변수는 최종 성능을 일관되게 개선하지 못했습니다. 기존 원본 피처인 최근 로그인 공백, 시청 시간, 완료율, 추천 클릭률 등이 이미 강한 이탈 신호를 담고 있었기 때문입니다.

따라서 최종 운영 모델은 원본 피처 기반 Stacking을 유지했습니다. 다만 파생변수와 논문 기반 proxy feature는 고위험 고객의 행동을 설명하고 캠페인 액션을 설계하는 보조 근거로 활용했습니다.

## Business Interpretation

이 모델은 모든 고객을 완벽하게 churn / non-churn으로 분류하기 위한 모델이 아닙니다. 실제 목적은 이탈 방지 캠페인의 우선순위를 정하는 것입니다.

Risk group 운영 방향:

| Group | Interpretation | Action |
| --- | --- | --- |
| High Risk | 즉시 개입 대상 | 할인, 업그레이드, 개인화 추천, 복귀 메시지 |
| Medium Risk | 저비용 개입 대상 | 콘텐츠 알림, 이어보기 유도, 가벼운 혜택 |
| Low Risk | 고비용 캠페인 제외 대상 | 일반 모니터링 |

고위험 고객은 다음 특성을 보였습니다.

- 최근 로그인 공백이 길다.
- 주간 시청 시간이 낮다.
- completion rate와 recommendation click rate가 낮다.
- Basic 요금제와 Mobile 사용 비중이 높다.

핵심 타겟은 상위 10% risk 고객 중 규모와 이탈률이 모두 큰 **성장기/장기 Basic 고객**입니다.

- 대상 규모: 586명
- 상위 10% 고객 중 비중: 약 58.6%
- 실제 churn rate: 약 88.6%
- 상위 10% 내 실제 이탈자 포함 비중: 약 59.2%

제안 액션:

- Mobile 사용 비중이 높은 가격 민감 고객에게는 **Mobile-only 저가 요금제**를 제안해 완전 해지를 저가 유지로 전환합니다.
- 성장기/장기 Basic 고객 중 risk score가 높은 고객에게는 **Standard 업그레이드 1개월 쿠폰**을 제공해 상위 요금제 가치를 체험하게 합니다.

외부 리텐션 사례와도 방향이 일치합니다. Harvard Business Review는 모든 고객을 동일하게 유지하려 하기보다 장기 가치가 높은 고객을 구분해 관리하는 것이 중요하다고 설명합니다. 고객 유지 관련 통계에서도 개인화된 경험과 충성도 프로그램, 고객 경험 개선이 이탈 방지에 중요한 요소로 제시됩니다.

참고 자료:

- HBR: https://hbr.org/2014/10/the-value-of-keeping-the-right-customers
- QR TIGER: https://www.qrcode-tiger.com/ko/customer-retention-statistics

## KKBOX Validation

Netflix-style 데이터는 합성 데이터이므로 모델 성능에 한계가 있었습니다. 이를 보완하기 위해 실제 구독 서비스 데이터인 KKBOX에 동일한 Stacking 파이프라인을 적용했습니다.

검증 결과:

| Experiment | Interpretation |
| --- | --- |
| Netflix -> KKBOX transfer | 도메인 차이로 인해 성능이 제한적 |
| KKBOX in-domain | ROC AUC 약 0.977, F1 약 0.881 |

이 결과는 Netflix-style 합성 데이터에서의 낮은 성능이 파이프라인 자체의 실패라기보다 데이터 특성과 도메인 차이에 의한 한계였음을 보여줍니다. 실제 기업 데이터로 학습과 평가를 진행하면 유의미한 이탈 예측과 리텐션 전략 수립이 가능하다는 근거로 해석했습니다.

## Notes

- `05_churn_modeling_tuning_ensemble.ipynb`는 최종 모델과 risk score를 저장합니다.
- `06_churn_model_interpretation_action.ipynb`는 저장된 결과를 읽어 해석과 액션 전략을 만듭니다.
- `07-*` 노트북은 추가 feature engineering / optimization 실험 기록입니다.
- `08_kkbox_transfer_indomain.ipynb`는 KKBOX 검증용 노트북입니다.
- `report/`는 현재 정적 HTML 리포트의 실제 소스입니다.
- `scripts/` 디렉토리는 사용하지 않습니다.
- `.venv/`, `catboost_info/`, `.ipynb_checkpoints/`, `.DS_Store`, `.matplotlib-cache/` 등은 git 추적에서 제외합니다.
