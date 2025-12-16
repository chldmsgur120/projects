# 후행품목 무역량 예측 프로젝트(Dacon 경진대회)

품목 간 공행성(CCF)을 **value(무역량) 기준으로만 계산**하고  
그 결과를 활용하여 후행 품목의 **2025년 8월 무역량(value)** 을 예측하는 머신러닝 프로젝트

---
## 📑 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 구조](#2-데이터-구조)
   - [Train 데이터](#21-train-데이터-traincsv)
   - [Submission 데이터](#22-submission-데이터-sample_submissioncsv)
3. [데이터 전처리](#3-데이터-전처리)
   - [날짜 정규화](#31-날짜-정규화)
   - [Monthly DataFrame 구성](#32-monthly-dataframe-구성)
4. [공행성(CCF) 기반 선행–후행 구조 추출](#4-공행성ccf-기반-선행후행-구조-추출)
   - [모든 품목 조합에 대한 CCF 계산](#41-모든-품목-조합ij에-대해-수행)
   - [relations_df 구성](#42-relations_df-구성)
5. [Feature Engineering (value 기반)](#5-feature-engineering-value-기반)
   - [Follower 기준 학습 데이터 구성](#51-follower-기준-학습-데이터-구성)
   - [예측 시점 Feature 구성](#52-예측-시점-feature)
6. [모델링 (value 기반 RandomForest)](#6-모델링-value-기반-randomforest)
   - [알고리즘](#61-알고리즘)
   - [학습 방식](#62-학습-방식)
   - [예측](#63-예측)
7. [제출 파일 생성](#7-제출-파일-생성)
   - [sample_submission 병합](#71-sample_submission-병합)
   - [relations_df 기반 필터링](#72-relations_df-기반-필터링)
   - [최종 결과](#73-최종-결과)
8. [평가지표](#평가지표)
9. [후행품목 무역량 예측 결과 해석 & 인사이트](#후행품목-무역량-예측-결과-해석--인사이트)
10. [한계점 및 개선 방향](#한계점-및-개선-방향)

---

## 1. 프로젝트 개요

이 프로젝트는 품목들 사이에 **무역량(value)** 변화가 서로 영향을 주고받는다는 가정에서 출발합니다.  
이를 위해 다음 과정을 수행했습니다:

- 모든 item_id 조합에 대해 **value 시계열을 사용**해 CCF(Cross-Correlation Function)를 계산하여  
  선행–후행(leader → follower) 구조 파악  
- 도출된 선후행 관계만을 사용하여 follower의 **다음 달 value** 예측

weight와 quantity 변수도 포함하여 분석을 진행하려 했으나,
데이터 검토 과정에서 두 변수 모두 값의 왜곡·비정상적 측정이 다수 확인되었습니다.
이에 따라 분석 신뢰도를 위해 해당 변수들은 전 단계에서 제외하였음

CCF 탐색, 모델 학습, 8월 예측까지 **모두 value 하나로만 진행한 프로젝트**입니다.

---

## 2. 데이터 구조

### 2.1 Train 데이터 (`train.csv`)

| 컬럼명   | 타입     | 설명 |
|---------|----------|------|
| item_id | category | 품목 식별자 (총 100개 품목) |
| year    | int      | 연도 (2022 ~ 2025) |
| month   | int      | 월 (1 ~ 12) |
| seq     | float    | 월 내 순서 (1.0, 2.0, 3.0) |
| type    | int      | 데이터 타입 구분 (본 데이터에서는 1로 고정) |
| hs4     | int      | HS 4단위 품목 코드 |
| weight  | float    | 무게 (데이터 품질 문제로 분석에서는 미사용) |
| quantity| float    | 수량 (데이터 품질 문제로 분석에서는 미사용) |
| value   | float    | 무역액(금액) – 본 프로젝트에서 유일하게 사용한 지표 |

---

### 2.2 Submission 데이터 (`sample_submission.csv`)

| 컬럼명          | 타입     | 설명 |
|----------------|----------|------|
| leading_item_id  | category | 선행(leader) 품목의 item_id |
| following_item_id| category | 후행(follower) 품목의 item_id |
| value            | int      | 예측해야 할 무역량(value). 초기값은 999999999로 채워져 있으며, 모델 예측값으로 대체하여 제출 |

---

## 3. 데이터 전처리

### 3.1 날짜 정규화
- year + month → 하나의 월 단위 date로 병합  
- seq는 월 단위 분석 기준과 맞지 않아 제거  
- value(무역량)만 남겨 item_id별 월 단위 시계열 정렬

### 3.2 monthly dataframe 구성
- item_id별로 `date`–`value`만 존재하는 단일 시계열 생성  
- 이후 모든 공행성 계산 및 예측 모델의 기반 데이터가 됨

---

## 4. 공행성(CCF) 기반 선행–후행 구조 추출

### 4.1 모든 품목 조합(i, j)에 대해 수행
- A.value(t)와 B.value(t) 간의 CCF 계산  
- lag 범위 내에서 **최대 상관도가 발생하는 lag 추출**
- **상관도의 절댓값이 0.3 이상인것을 공행성이 있다고 판단**
- **lag > 0** → A의 변화가 B보다 먼저 나타나므로 A는 **leader**, B는 **follower**로 해석  
- **lag < 0** → A의 변화가 B보다 늦게 나타나므로 A는 **follower**, B는 **leader**로 해석

### 4.2 relations_df 구성
- 컬럼: `leader`, `follower`, `lag`
- 의미: follower는 leader의 value(t - lag)를 따라가는 구조
- 전 과정은 **value 단일 변수 기반의 공행성 구조**

---

## 5. Feature Engineering (value 기반)

모든 feature는 **오직 value만 사용**하여 생성되었습니다.

### 5.1 follower 기준 학습 데이터 구성
각 follower(item_id)에 대해:

- target = follower.value(t)  
- feature = follower를 따라가는 leader들의  
  - lag-shift된 leader.value(t - lag)

즉, value 하나만 사용한 단순하지만 명확한 구조입니다.

### 5.2 예측 시점 feature
2025년 8월 예측 시:

- leader.value(2025-08 - lag)를 추출  
- 이 값들만으로 X_pred 구성 (value-only feature set)

---

## 6. 모델링 (value 기반 RandomForest)

### 6.1 알고리즘
사용된 모델은 다음과 같습니다:

- **RandomForestRegressor**
  - n_estimators = 500
  - max_depth = 15
  - n_jobs = -1
  - random_state = 42

### 6.2 학습 방식
- follower마다 개별 RandomForest 모델을 생성하여 value(t) 예측  
- value 이외의 변수는 사용하지 않음  
- feature 개수는 follower가 가진 leader 수에 따라 달라짐  
- 학습 가능한 데이터 수가 부족한 follower는 자동 스킵

### 6.3 예측
- 기준 날짜: **2025-08-31**
- follower에 대한 value 예측 생성  
- 결과는 `(following_item_id, forecast_month, pred_value)` 형태로 저장됨

---

## 7. 제출 파일 생성

### 7.1 sample_submission 병합
- following_item_id 기준으로 pred_value를 value로 대체  
- leading_item_id는 유지  
- NaN 제거 후 약 **2,387개 행** 유지

### 7.2 relations_df 기반 필터링
- 실제 공행성 관계가 존재하는 (leader, follower) 조합만 남김  
- lag는 불필요하므로 최종 출력에서 제거

### 7.3 최종 결과
- **value 예측치만 포함된 submission.csv 생성**  
- 모든 예측은 **value 시계열 기반 모델의 출력**

---

## 평가지표
•	평가 산식 : Score = 0.6 × F1 + 0.4 × (1 − NMAE)
1) F1 = (2 × Precision × Recall) ÷ (Precision + Recall)
- Precision = TP ÷ (TP + FP)
- Recall = TP ÷ (TP + FN)

여기서
- TP(True Positive): 정답과 예측 모두에 포함된 공행성쌍
- FP(False Positive): 예측에는 있으나 정답에는 없는 쌍
- FN(False Negative): 정답에는 있으나 예측에 없는 쌍

3) NMAE = (1 / |U|) × Σ[min(1, |y_true - y_pred| ÷ (|y_true| + ε))]
- U = 정답 쌍(G)과 예측 쌍(P)의 합집합
- y_true: 정답의 다음달 무역량 (정수 변환)
- y_pred: 예측 무역량 (정수 반올림)
- FN 또는 FP에 해당하는 경우 오차 1.0(100%, 최하점)로 처리
- 오차가 100%를 초과하는 경우에도 1.0(100%, 최하점)로 처리

---

---

## 후행품목 무역량 예측 결과 해석 & 인사이트

### 결과 해석 및 인사이트

본 프로젝트에서는  
**공행성(선·후행) 쌍 식별**과 **후행 품목의 무역량 예측**을 함께 수행하는 구조의 대회 과제를 해결하였다.

모델링 과정에서 다음과 같은 설정에서 가장 높은 성능을 확인하였다.

- 시차 상관 분석 결과  
  - **최대 lag = 6**일 때 공행성 관계 식별 성능이 가장 안정적으로 나타남
- 공행성 쌍 필터링 기준  
  - 상관계수의 **절댓값 |corr| ≥ 0.33** 이상인 경우만 공행성 후보로 선정
- 후행 무역량 예측 모델  
  - Random Forest
    - `max_depth = 15`
    - `n_estimators = 500`

해당 설정을 적용한 결과,
- 초기 점수 **0.2896**
- 최종 점수 **0.3405**

까지 성능을 개선할 수 있었으며,  
공행성 식별 기준과 후행 예측 모델의 복잡도 간 균형이  
전체 평가 점수에 중요한 영향을 미친다는 점을 확인하였다.

---

## 한계점 및 개선 방향

### 한계점

본 대회는  
**공행성(선·후행) 쌍 식별** 과  
**후행 품목 무역량 예측** 을 동시에 평가하는 구조로 설계되어 있어,

- 별도의 검증 데이터를 분리한 오프라인 평가가 어려웠고
- `submit.csv 제출 → 점수 확인` 방식으로만 모델 성능을 판단할 수 있었다.

또한,
- 하루 제출 횟수 제한으로 인해
  - 다양한 모델 구조 및 하이퍼파라미터를 충분히 탐색하지 못한 한계가 존재한다.


### 개선 방향

보다 안정적인 모델 검증과 성능 향상을 위해  
다음과 같은 방식의 개선이 가능할 것으로 판단된다.

#### 1) 후행 무역량 예측 모델 검증 구조 개선
- 전체 시계열 중 **마지막 N개월을 검증 데이터로 분리**
- 공행성 쌍이라고 가정한 (leader, follower) pair에 대해서만
  - `value(무역량)` 예측 모델을 학습·검증
- 이를 통해
  - submit 의존도를 줄이고
  - 후행 예측 모델 자체의 성능을 기준으로 보다 정교한 튜닝 가능

#### 2) 공행성 쌍 식별을 이진 분류 문제로 분리
- 마지막 N개월을 기준으로
  - 특정 시점에서 상관이 크면 `1`
  - 그렇지 않으면 `0`
  으로 라벨링
- 과거 구간에서 학습한 공행성 패턴이
  - 마지막 N개월 구간에서도 실제로 동행·후행 움직임을 보이는지 검증
- 이후
  - 공행성 분류 모델과
  - 후행 무역량 예측 모델을 단계적으로 결합

이와 같은 구조를 적용할 경우,  
현재의 submit 중심 접근 방식보다  
**모델 해석력과 일반화 성능이 모두 향상된 결과**를 도출할 수 있을 것으로 기대된다.


