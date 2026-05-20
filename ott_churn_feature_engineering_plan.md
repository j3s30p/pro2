# OTT 구독 해지 예측 Feature Engineering Plan

## 0. 목적

이 문서는 Netflix OTT 구독 해지(`churned`) 예측 모델링을 위해 사용할 피쳐 엔지니어링 기준서다.

현재 데이터셋은 Netflix와 같은 **글로벌 SVOD** 서비스 이용자 데이터이며, 논문 요약을 바탕으로 다음 3가지 축을 중심으로 파생변수를 생성한다.

1. **가격 대비 이용 가치**
2. **추천 시스템 / 개인화 반응**
3. **멀티호밍·전환 위험 / 서비스 유지 필요성**

---

## 1. 데이터셋 기본 컬럼

| 컬럼명 | 설명 |
|---|---|
| `user_id` | 사용자 고유 식별자 |
| `age` | 사용자 나이 |
| `gender` | 사용자 성별 |
| `region` | 사용자 지역 |
| `subscription_type` | 구독 요금제 유형 |
| `payment_method` | 결제 수단 |
| `primary_device` | 주 사용 시청 기기 |
| `account_age_months` | 계정 생성 후 경과 개월 수 |
| `favorite_genre` | 선호 장르 |
| `time_of_day` | 주 이용 시간대 |
| `recommendation_source` | 추천 또는 유입 경로 |
| `session_count` | 집계 기간 내 접속 세션 수 |
| `avg_watch_time_minutes_per_week` | 주당 평균 시청 시간 |
| `watch_sessions_per_week` | 주당 시청 세션 수 |
| `completion_rate` | 콘텐츠 평균 완료율 |
| `avg_rating_given` | 사용자가 남긴 평균 평점 |
| `app_rating` | 앱에 대한 사용자 평가 점수 |
| `recommendation_click_rate` | 추천 콘텐츠 클릭률 |
| `days_since_last_login` | 마지막 로그인 이후 경과 일수 |
| `churned` | 이탈 여부 타겟값. 1 = 이탈, 0 = 유지 |

---

## 2. 논문 기반 Feature Engineering 방향

### 2.1 글로벌 SVOD 기준 핵심 해지 요인

Netflix는 글로벌 SVOD로 간주한다. 관련 논문 결과에서 글로벌 SVOD의 해지에는 다음 요인이 중요했다.

| 요인 | 방향 | 해석 |
|---|---|---|
| 요금 부담 | churn 증가 | 가격 부담이 클수록 해지 가능성 증가 |
| 추천시스템 불만족 | churn 증가 | 추천 경험이 나쁠수록 해지 가능성 증가 |
| OTT 멀티호밍 | churn 증가 | 여러 OTT를 같이 쓰는 사용자는 특정 서비스 해지 가능성 증가 |
| 주 시청기기 TV | churn 증가 | TV 중심 이용자는 글로벌 SVOD 해지 확률이 높게 나타남 |
| 드라마 선호 | churn 감소 | 글로벌 SVOD에서 유지 가능성과 관련 |
| 예능·오락 선호 | churn 감소 | 글로벌 SVOD에서 유지 가능성과 관련 |

핵심 메시지:

> Netflix 같은 글로벌 SVOD의 churn 예측에서는 콘텐츠 부족 자체보다 가격 대비 가치, 추천 경험, 이용 강도, 대체 가능성/전환 위험이 더 중요하다.

---

## 3. Feature Engineering 원칙

### 3.1 직접 컬럼이 없는 개념은 proxy로 만든다

현재 데이터셋에는 아래 컬럼이 직접 존재하지 않는다.

- 실제 월 구독료
- 다른 OTT 동시 구독 여부
- 실제 추천 만족도 설문
- 동일 시리즈 연속 시청 로그
- 검색 실패, 자막 오류, 끊김 등 기술 문제 로그

따라서 아래와 같이 대체 피쳐를 만든다.

| 개념 | 직접 컬럼 | 대체 proxy |
|---|---|---|
| 가격 부담 | 없음 | `subscription_type`, 이용량 대비 요금제 |
| 추천 만족도 | 없음 | 추천 클릭률, 완료율, 평균 평점 |
| 멀티호밍 | 없음 | 낮은 이용량 + 낮은 추천 반응 + 긴 미접속 |
| 구독 피로감 | 없음 | 장기 가입 + 최근 이용 감소 |
| 몰아보기 | 없음 | 세션당 평균 시청 시간 + 완료율 |

---

## 4. 기본 전처리

### 4.1 라이브러리

```python
import numpy as np
import pandas as pd
```

### 4.2 요금제 등급화

`subscription_type`은 실제 가격 컬럼이 아니므로 요금제 수준을 나타내는 ordinal proxy로 변환한다.

```python
plan_map = {
    "Basic": 1,
    "Standard": 2,
    "Premium": 3
}

df["plan_tier"] = df["subscription_type"].map(plan_map)
```

### 4.3 결측 매핑 확인

```python
print(df["subscription_type"].value_counts(dropna=False))
print(df["plan_tier"].isna().sum())
```

`plan_tier`에 결측이 생기면 실제 값 이름을 확인한 뒤 `plan_map`을 수정한다.

---

## 5. RFM형 OTT 피쳐

일반 RFM을 OTT 서비스에 맞게 변형한다.

| RFM | 일반 의미 | OTT 변환 |
|---|---|---|
| Recency | 최근 구매 시점 | 마지막 로그인 이후 경과일 |
| Frequency | 구매 빈도 | 주간 시청 세션 수 / 접속 세션 수 |
| Monetary | 구매 금액 | 요금제 등급 |

```python
eps = 1e-6

df["recency_score"] = df["days_since_last_login"]
df["frequency_score"] = df["watch_sessions_per_week"]
df["monetary_score"] = df["plan_tier"]

df["rfm_churn_risk"] = (
    df["recency_score"] * 0.5
    - df["frequency_score"] * 0.3
    + df["monetary_score"] * 0.2
)
```

해석:

> 마지막 접속일이 오래됐고, 주간 시청 빈도는 낮은데, 비싼 요금제를 쓰고 있다면 해지 위험이 높을 수 있다.

주의:

- 이 가중치 `0.5`, `0.3`, `0.2`는 임시 가중치다.
- 실제 성능 비교 시에는 이 수동 점수보다 개별 피쳐들을 모델에 넣는 방식이 더 안정적일 수 있다.

---

## 6. 가격 대비 이용 가치 피쳐

논문에서 글로벌 SVOD의 핵심 해지 요인 중 하나는 요금 부담이다.  
따라서 사용자가 지불하는 요금제 수준 대비 얼마나 서비스를 이용하는지를 나타내는 피쳐를 만든다.

```python
df["watch_time_per_plan"] = (
    df["avg_watch_time_minutes_per_week"] / (df["plan_tier"] + eps)
)

df["sessions_per_plan"] = (
    df["watch_sessions_per_week"] / (df["plan_tier"] + eps)
)

df["completion_per_plan"] = (
    df["completion_rate"] / (df["plan_tier"] + eps)
)

df["price_burden_proxy"] = (
    df["plan_tier"] / np.log1p(df["avg_watch_time_minutes_per_week"])
)

df["high_plan_low_usage"] = (
    (df["plan_tier"] >= 3) &
    (df["avg_watch_time_minutes_per_week"] < df["avg_watch_time_minutes_per_week"].median())
).astype(int)

df["value_for_money_score"] = (
    df["avg_watch_time_minutes_per_week"]
    + df["watch_sessions_per_week"] * 10
    + df["completion_rate"]
) / (df["plan_tier"] + eps)
```

### 주요 피쳐 해석

| 피쳐 | 해석 |
|---|---|
| `watch_time_per_plan` | 요금제 등급 대비 시청 시간 |
| `sessions_per_plan` | 요금제 등급 대비 시청 빈도 |
| `completion_per_plan` | 요금제 등급 대비 콘텐츠 완료율 |
| `price_burden_proxy` | 시청 시간이 낮은 고요금제 사용자일수록 커짐 |
| `high_plan_low_usage` | Premium인데 시청 시간이 낮은 사용자 |
| `value_for_money_score` | 가격 대비 이용 가치 종합 점수 |

---

## 7. 추천 시스템 / 개인화 피쳐

논문에서 글로벌 SVOD는 추천시스템 불만족이 해지와 관련이 있었다.  
현재 데이터에는 직접적인 추천 만족도 설문이 없으므로, 추천 클릭률과 콘텐츠 완료율, 평점을 사용해 proxy를 만든다.

```python
df["recommendation_effectiveness"] = (
    df["recommendation_click_rate"] * df["completion_rate"]
)

df["recommendation_mismatch"] = (
    df["recommendation_click_rate"] - df["completion_rate"]
)

df["personalization_risk"] = (
    (df["recommendation_click_rate"] < df["recommendation_click_rate"].median()) &
    (df["app_rating"] < df["app_rating"].median())
).astype(int)

df["is_algorithm_source"] = (
    df["recommendation_source"] == "Algorithm"
).astype(int)

df["recommendation_satisfaction_proxy"] = (
    df["recommendation_click_rate"]
    + df["completion_rate"]
    + df["avg_rating_given"]
) / 3
```

### 주요 피쳐 해석

| 피쳐 | 해석 |
|---|---|
| `recommendation_effectiveness` | 추천 클릭 후 실제 소비까지 이어지는 정도 |
| `recommendation_mismatch` | 추천은 눌렀지만 만족 소비로 이어지지 않았을 가능성 |
| `personalization_risk` | 추천 반응과 앱 평가가 모두 낮은 사용자 |
| `is_algorithm_source` | 알고리즘 추천 기반 유입 여부 |
| `recommendation_satisfaction_proxy` | 추천 경험 만족도 대체 점수 |

주의:

> `recommendation_source != "Algorithm"`이라고 해서 추천 불만족이라고 단정하면 안 된다.  
> 안전한 해석은 "알고리즘 추천 영향력이 낮은 사용자"다.

---

## 8. 멀티호밍·전환 위험 proxy

현재 데이터셋에는 다른 OTT 동시 구독 여부가 없다.  
따라서 직접적인 멀티호밍 피쳐는 만들 수 없고, Netflix를 유지할 필요성이 낮아 보이는 패턴을 proxy로 만든다.

```python
df["switching_risk_proxy"] = (
    (df["days_since_last_login"] > df["days_since_last_login"].median()) &
    (df["avg_watch_time_minutes_per_week"] < df["avg_watch_time_minutes_per_week"].median()) &
    (df["recommendation_click_rate"] < df["recommendation_click_rate"].median())
).astype(int)

df["service_dependency_score"] = (
    df["watch_sessions_per_week"] * 0.4
    + df["session_count"] * 0.3
    + df["avg_watch_time_minutes_per_week"] * 0.3
)

df["low_service_need"] = (
    df["service_dependency_score"] < df["service_dependency_score"].median()
).astype(int)
```

### 주요 피쳐 해석

| 피쳐 | 해석 |
|---|---|
| `switching_risk_proxy` | 오래 미접속 + 낮은 시청량 + 낮은 추천 반응 |
| `service_dependency_score` | 서비스 의존도 |
| `low_service_need` | 서비스 필요성이 낮은 사용자 |

---

## 9. 구독 피로감 proxy

구독 피로감은 여러 구독 서비스 이용, 과도한 선택, 구독 관리 부담에서 발생하는 심리적 피로 상태다.  
현재 데이터에는 설문 문항이 없으므로 장기 가입자 중 최근 이용 강도가 낮아진 사용자를 proxy로 잡는다.

```python
df["subscription_fatigue_proxy"] = (
    (df["account_age_months"] > df["account_age_months"].median()) &
    (df["days_since_last_login"] > df["days_since_last_login"].median()) &
    (df["watch_sessions_per_week"] < df["watch_sessions_per_week"].median())
).astype(int)

df["usage_per_account_age"] = (
    df["avg_watch_time_minutes_per_week"] / (df["account_age_months"] + 1)
)
```

### 주요 피쳐 해석

| 피쳐 | 해석 |
|---|---|
| `subscription_fatigue_proxy` | 오래 구독했지만 최근 이용이 낮은 사용자 |
| `usage_per_account_age` | 계정 연령 대비 현재 이용 강도 |

---

## 10. 몰아보기 / 콘텐츠 고갈 proxy

논문에서 몰아보기는 동일 프로그램 에피소드 2편 이상을 지속적으로 보는 것으로 정의된다.  
그러나 현재 데이터에는 에피소드 단위 로그가 없으므로 정확한 몰아보기 피쳐는 만들 수 없다.  
대신 세션당 평균 시청 시간과 완료율을 이용해 약한 proxy를 만든다.

```python
df["avg_minutes_per_watch_session"] = (
    df["avg_watch_time_minutes_per_week"] / (df["watch_sessions_per_week"] + eps)
)

df["binge_intensity_proxy"] = (
    df["avg_minutes_per_watch_session"] * df["completion_rate"]
)

df["content_exhaustion_proxy"] = (
    (df["avg_watch_time_minutes_per_week"] > df["avg_watch_time_minutes_per_week"].quantile(0.75)) &
    (df["completion_rate"] > df["completion_rate"].quantile(0.75))
).astype(int)
```

### 주요 피쳐 해석

| 피쳐 | 해석 |
|---|---|
| `avg_minutes_per_watch_session` | 한 시청 세션당 평균 시청 시간 |
| `binge_intensity_proxy` | 몰아보기 가능성 대체 점수 |
| `content_exhaustion_proxy` | 많이 보고 완료율도 높은 사용자 |

주의:

> 이 피쳐는 실제 몰아보기를 직접 측정하지 않는다.  
> 정확한 몰아보기 분석에는 시리즈 ID, 에피소드 ID, 시청 순서, 시청 시간 로그가 필요하다.

---

## 11. 장르 / 디바이스 피쳐

논문 결과에서 글로벌 SVOD 기준으로 TV 주시청, 드라마 선호, 예능·오락 선호가 유의했다.  
현재 데이터셋의 실제 범주값에 맞춰 매핑해야 한다.

```python
df["is_tv_device"] = (
    df["primary_device"] == "Smart TV"
).astype(int)

df["is_retention_genre"] = (
    df["favorite_genre"].isin(["Drama", "Comedy"])
).astype(int)
```

주의:

- 논문에서 예능·오락은 현재 데이터셋에서 `Comedy`에 가까운 범주로 볼 수 있다.
- 실제 `favorite_genre` 값 목록을 확인한 뒤 매핑을 조정한다.

```python
print(df["primary_device"].value_counts())
print(df["favorite_genre"].value_counts())
```

---

## 12. 최종 추천 파생변수 목록

```python
new_features = [
    "plan_tier",

    # RFM
    "recency_score",
    "frequency_score",
    "monetary_score",
    "rfm_churn_risk",

    # 가격 대비 이용 가치
    "watch_time_per_plan",
    "sessions_per_plan",
    "completion_per_plan",
    "price_burden_proxy",
    "high_plan_low_usage",
    "value_for_money_score",

    # 추천 시스템 / 개인화
    "recommendation_effectiveness",
    "recommendation_mismatch",
    "personalization_risk",
    "is_algorithm_source",
    "recommendation_satisfaction_proxy",

    # 전환 위험 / 서비스 필요성
    "switching_risk_proxy",
    "service_dependency_score",
    "low_service_need",

    # 구독 피로감
    "subscription_fatigue_proxy",
    "usage_per_account_age",

    # 몰아보기 / 콘텐츠 고갈
    "avg_minutes_per_watch_session",
    "binge_intensity_proxy",
    "content_exhaustion_proxy",

    # 논문 기반 범주 피쳐
    "is_tv_device",
    "is_retention_genre",
]
```

---

## 13. 전체 피쳐 생성 함수

Codex에서는 아래 함수를 기준으로 `feature_engineering.py` 또는 노트북 셀에 구현한다.

```python
import numpy as np
import pandas as pd


def create_ott_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Netflix OTT churn 예측용 파생변수 생성 함수.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임

    Returns
    -------
    pd.DataFrame
        파생변수가 추가된 데이터프레임
    """
    df = df.copy()
    eps = 1e-6

    # 1. 요금제 등급화
    plan_map = {
        "Basic": 1,
        "Standard": 2,
        "Premium": 3,
    }
    df["plan_tier"] = df["subscription_type"].map(plan_map)

    # 매핑되지 않은 값이 있으면 중앙값으로 임시 대체
    # 실제 프로젝트에서는 value_counts()로 원인을 확인하고 plan_map을 수정하는 것이 우선
    if df["plan_tier"].isna().any():
        df["plan_tier"] = df["plan_tier"].fillna(df["plan_tier"].median())

    # 2. RFM형 OTT 피쳐
    df["recency_score"] = df["days_since_last_login"]
    df["frequency_score"] = df["watch_sessions_per_week"]
    df["monetary_score"] = df["plan_tier"]

    df["rfm_churn_risk"] = (
        df["recency_score"] * 0.5
        - df["frequency_score"] * 0.3
        + df["monetary_score"] * 0.2
    )

    # 3. 가격 대비 이용 가치
    df["watch_time_per_plan"] = (
        df["avg_watch_time_minutes_per_week"] / (df["plan_tier"] + eps)
    )

    df["sessions_per_plan"] = (
        df["watch_sessions_per_week"] / (df["plan_tier"] + eps)
    )

    df["completion_per_plan"] = (
        df["completion_rate"] / (df["plan_tier"] + eps)
    )

    df["price_burden_proxy"] = (
        df["plan_tier"] / np.log1p(df["avg_watch_time_minutes_per_week"])
    )

    df["high_plan_low_usage"] = (
        (df["plan_tier"] >= 3)
        & (df["avg_watch_time_minutes_per_week"] < df["avg_watch_time_minutes_per_week"].median())
    ).astype(int)

    df["value_for_money_score"] = (
        df["avg_watch_time_minutes_per_week"]
        + df["watch_sessions_per_week"] * 10
        + df["completion_rate"]
    ) / (df["plan_tier"] + eps)

    # 4. 추천 시스템 / 개인화
    df["recommendation_effectiveness"] = (
        df["recommendation_click_rate"] * df["completion_rate"]
    )

    df["recommendation_mismatch"] = (
        df["recommendation_click_rate"] - df["completion_rate"]
    )

    df["personalization_risk"] = (
        (df["recommendation_click_rate"] < df["recommendation_click_rate"].median())
        & (df["app_rating"] < df["app_rating"].median())
    ).astype(int)

    df["is_algorithm_source"] = (
        df["recommendation_source"] == "Algorithm"
    ).astype(int)

    df["recommendation_satisfaction_proxy"] = (
        df["recommendation_click_rate"]
        + df["completion_rate"]
        + df["avg_rating_given"]
    ) / 3

    # 5. 멀티호밍·전환 위험 proxy
    df["switching_risk_proxy"] = (
        (df["days_since_last_login"] > df["days_since_last_login"].median())
        & (df["avg_watch_time_minutes_per_week"] < df["avg_watch_time_minutes_per_week"].median())
        & (df["recommendation_click_rate"] < df["recommendation_click_rate"].median())
    ).astype(int)

    df["service_dependency_score"] = (
        df["watch_sessions_per_week"] * 0.4
        + df["session_count"] * 0.3
        + df["avg_watch_time_minutes_per_week"] * 0.3
    )

    df["low_service_need"] = (
        df["service_dependency_score"] < df["service_dependency_score"].median()
    ).astype(int)

    # 6. 구독 피로감 proxy
    df["subscription_fatigue_proxy"] = (
        (df["account_age_months"] > df["account_age_months"].median())
        & (df["days_since_last_login"] > df["days_since_last_login"].median())
        & (df["watch_sessions_per_week"] < df["watch_sessions_per_week"].median())
    ).astype(int)

    df["usage_per_account_age"] = (
        df["avg_watch_time_minutes_per_week"] / (df["account_age_months"] + 1)
    )

    # 7. 몰아보기 / 콘텐츠 고갈 proxy
    df["avg_minutes_per_watch_session"] = (
        df["avg_watch_time_minutes_per_week"] / (df["watch_sessions_per_week"] + eps)
    )

    df["binge_intensity_proxy"] = (
        df["avg_minutes_per_watch_session"] * df["completion_rate"]
    )

    df["content_exhaustion_proxy"] = (
        (df["avg_watch_time_minutes_per_week"] > df["avg_watch_time_minutes_per_week"].quantile(0.75))
        & (df["completion_rate"] > df["completion_rate"].quantile(0.75))
    ).astype(int)

    # 8. 장르 / 디바이스
    df["is_tv_device"] = (
        df["primary_device"] == "Smart TV"
    ).astype(int)

    df["is_retention_genre"] = (
        df["favorite_genre"].isin(["Drama", "Comedy"])
    ).astype(int)

    return df
```

---

## 14. 생성 후 점검 코드

```python
df_fe = create_ott_churn_features(df)

# 생성된 피쳐 확인
print(df_fe.shape)
print(df_fe[new_features].head())

# 결측치 확인
missing = df_fe[new_features].isna().sum()
print(missing[missing > 0])

# 무한대 값 확인
inf_count = np.isinf(df_fe[new_features].select_dtypes(include=[np.number])).sum()
print(inf_count[inf_count > 0])

# 타겟별 중앙값 확인
check_cols = [
    "days_since_last_login",
    "avg_watch_time_minutes_per_week",
    "watch_sessions_per_week",
    "recommendation_click_rate",
    "completion_rate",
    "value_for_money_score",
    "switching_risk_proxy",
]

print(df_fe.groupby("churned")[check_cols].median())
```

---

## 15. 모델링 시 주의사항

### 15.1 Target leakage 확인

아래 피쳐들은 churn 이후에 기록된 값이면 누수가 될 수 있다.

- `days_since_last_login`
- `app_rating`
- `avg_rating_given`
- `recommendation_click_rate`
- `completion_rate`

현재 데이터셋이 특정 기준 시점 이전의 행동 데이터로 구성되어 있다면 사용 가능하다.  
하지만 churn 이후의 행동까지 포함되어 있다면 target leakage가 발생할 수 있다.

### 15.2 점수형 피쳐는 스케일링 고려

Logistic Regression, SVM, KNN 등 거리/계수 기반 모델에서는 스케일링이 필요하다.  
LightGBM, XGBoost, RandomForest 같은 트리 기반 모델에서는 필수는 아니다.

### 15.3 수동 점수와 원본 피쳐를 모두 넣을 때 중복성 확인

예를 들어 아래 피쳐들은 서로 강하게 연관될 수 있다.

- `value_for_money_score`
- `watch_time_per_plan`
- `avg_watch_time_minutes_per_week`
- `plan_tier`

모델 성능은 좋아질 수 있지만, 해석은 복잡해질 수 있다.  
Feature importance, permutation importance, SHAP 등을 통해 확인한다.

---

## 16. 실험 추천 순서

### Step 1. Baseline

원본 피쳐만 사용한다.

```python
baseline_features = [
    col for col in df.columns
    if col not in ["user_id", "churned"]
]
```

### Step 2. Feature Engineering 추가

```python
df_fe = create_ott_churn_features(df)

model_features = [
    col for col in df_fe.columns
    if col not in ["user_id", "churned"]
]
```

### Step 3. 성능 비교

비교 지표는 다음을 우선한다.

- PR-AUC
- F1-score
- Recall
- Precision
- ROC-AUC
- Lift@10%
- Lift@20%

churn 예측에서는 accuracy보다 PR-AUC, F1, Recall, Lift가 더 중요하다.

### Step 4. Threshold 조정

모델 확률값에서 기본 threshold 0.5만 사용하지 말고, F1 또는 비즈니스 목적에 맞춰 threshold를 조정한다.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_valid, y_pred_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = f1_scores.argmax()

best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)
```

---

## 17. 발표용 한 줄 설명

### 가격 대비 이용 가치

> 논문에서 글로벌 SVOD 해지에는 요금 부담이 크게 작용했기 때문에, 요금제 등급 대비 실제 이용량을 나타내는 `value_for_money_score`, `watch_time_per_plan` 등을 생성했다.

### 추천 시스템

> 추천시스템 불만족이 글로벌 SVOD 해지와 관련이 있었기 때문에, 추천 클릭률과 완료율, 평점을 결합해 추천 만족도 proxy를 만들었다.

### 전환 위험

> 실제 멀티호밍 여부는 없지만, 낮은 이용량, 긴 미접속, 낮은 추천 반응을 결합해 Netflix 유지 필요성이 낮은 사용자를 나타내는 `switching_risk_proxy`를 생성했다.

### 구독 피로감

> 장기 가입자 중 최근 접속과 시청 빈도가 낮은 사용자는 구독 피로감 또는 서비스 필요성 감소가 나타났을 가능성이 있어 `subscription_fatigue_proxy`를 만들었다.

### 몰아보기

> 논문에서는 몰아보기가 콘텐츠 고갈과 해지로 이어질 수 있다고 보았으나, 현재 데이터에는 에피소드 로그가 없어 세션당 평균 시청 시간과 완료율로 약한 proxy를 생성했다.

---

## 18. Codex 작업 요청 문구 예시

Codex에 아래와 같이 요청한다.

```text
이 문서의 Feature Engineering Plan을 기준으로 현재 Netflix churn 데이터셋에 적용할 `create_ott_churn_features(df)` 함수를 구현해줘.

요구사항:
1. 원본 df를 직접 수정하지 말고 copy해서 반환해줘.
2. `subscription_type`을 `plan_tier`로 매핑해줘.
3. 문서의 파생변수를 모두 생성해줘.
4. 범주값이 예상과 다를 수 있으니 value_counts 확인 코드를 함께 넣어줘.
5. 생성된 피쳐 목록을 `new_features` 리스트로 관리해줘.
6. target leakage 위험이 있는 컬럼은 주석으로 표시해줘.
7. 생성 후 결측치와 무한대 값이 있는지 점검하는 코드를 추가해줘.
8. baseline 모델과 feature engineering 적용 모델의 PR-AUC, F1, ROC-AUC를 비교하는 실험 코드를 작성해줘.
```

---

## 19. 최종 요약

현재 데이터셋에서 가장 타당한 피쳐 엔지니어링 방향은 다음이다.

1. **RFM 변형**
   - Recency: `days_since_last_login`
   - Frequency: `watch_sessions_per_week`, `session_count`
   - Monetary: `subscription_type` → `plan_tier`

2. **가격 대비 가치**
   - `watch_time_per_plan`
   - `sessions_per_plan`
   - `price_burden_proxy`
   - `value_for_money_score`

3. **추천 시스템 반응**
   - `recommendation_effectiveness`
   - `recommendation_mismatch`
   - `personalization_risk`
   - `recommendation_satisfaction_proxy`

4. **전환 위험 / 서비스 필요성**
   - `switching_risk_proxy`
   - `service_dependency_score`
   - `low_service_need`

5. **구독 피로감 / 몰아보기 proxy**
   - `subscription_fatigue_proxy`
   - `usage_per_account_age`
   - `binge_intensity_proxy`
   - `content_exhaustion_proxy`

이 피쳐들은 단순히 많이 만드는 것이 목적이 아니라, 논문에서 확인된 글로벌 SVOD 해지 요인인 **요금 부담, 추천시스템 불만족, 멀티호밍/전환 가능성**을 현재 데이터셋에서 최대한 대체 표현하기 위한 것이다.
