# Task Type별 CSV 저장 형식

샘플별 평가 결과를 오프라인 평가용으로 저장하는 형식입니다.

## 파일 구조

```
results/
└── {YYYYMMDD}/
    ├── {HHMMSS}_{script_name}_classification.csv
    ├── {HHMMSS}_{script_name}_regression.csv
    ├── {HHMMSS}_{script_name}_molecule_generation.csv
    └── {HHMMSS}_{script_name}_captioning.csv
```

---

## Classification CSV

**파일명**: `{timestamp}_{script_name}_classification.csv`

**태스크**: bace, smol-property_prediction-bbbp, smol-property_prediction-clintox, smol-property_prediction-hiv, smol-property_prediction-sider

| 컬럼 | 타입 | 설명 |
|------|------|------|
| idx | int | 샘플 인덱스 |
| task | str | 태스크 이름 |
| label | str | raw 데이터 (예: `<BOOLEAN> True </BOOLEAN>`) |
| pred | int | 파싱된 예측 (0 or 1) |
| prob | float | 예측 확률 (positive class) |
| correct | int | 정답 여부 (1 or 0) |

**오프라인 평가 metric**:
- accuracy = mean(correct)
- f1, precision, recall = sklearn.metrics 사용
- roc_auc = sklearn.metrics.roc_auc_score(label_parsed, prob)
- roc_auc_all = 전체 샘플 포함 (파싱 실패도 포함)

**roc_auc vs roc_auc_all 차이**:
- `roc_auc`: 파싱 실패(prob=-1) 샘플을 **제외**하고 계산. 유효한 prob만 사용하여 모델의 순수 성능 측정.
- `roc_auc_all`: 파싱 실패 포함 **전체** 샘플로 계산. prob=-1인 샘플도 그대로 사용 (온라인 평가와 동일).
- `failure_rate`: 파싱 실패 비율 (prob < 0인 샘플 비율)

---

## Regression CSV

**파일명**: `{timestamp}_{script_name}_regression.csv`

**태스크**: aqsol-logS, qm9_homo, qm9_homo_lumo_gap, qm9_lumo, smol-property_prediction-esol, smol-property_prediction-lipo

| 컬럼 | 타입 | 설명 |
|------|------|------|
| idx | int | 샘플 인덱스 |
| task | str | 태스크 이름 |
| label | str | raw 데이터 (예: `<NUMBER> -0.2345 </NUMBER>`) |
| pred | float | 파싱된 예측값 |
| error | float | pred - target (부호 있는 오차) |

**오프라인 평가 metric**:
- mae = mean(|error|)
- mse = mean(error^2)
- rmse = sqrt(mse)
- failure_rate = pred가 NaN인 비율

---

## Molecule Generation CSV

**파일명**: `{timestamp}_{script_name}_molecule_generation.csv`

**태스크**: chebi-20-text2mol, forward_reaction_prediction, reagent_prediction, retrosynthesis, smol-forward_synthesis, smol-molecule_generation, smol-retrosynthesis

| 컬럼 | 타입 | 설명 |
|------|------|------|
| idx | int | 샘플 인덱스 |
| task | str | 태스크 이름 |
| label | str | raw 데이터 (SELFIES, 태그 포함) |
| pred | str | 파싱된 예측 (SMILES or SELFIES) |
| validity | int | 유효한 분자 여부 (1 or 0) |
| exact_match | int | 정확히 일치 여부 (1 or 0) |
| MACCS_FTS | float | MACCS fingerprint Tanimoto similarity |
| RDK_FTS | float | RDK fingerprint Tanimoto similarity |
| morgan_FTS | float | Morgan fingerprint Tanimoto similarity |
| levenshtein | int | Levenshtein distance |

**오프라인 평가 metric**:
- validity_ratio = mean(validity)
- exact_match_ratio = mean(exact_match)
- MACCS_FTS = mean(MACCS_FTS)
- RDK_FTS = mean(RDK_FTS)
- morgan_FTS = mean(morgan_FTS)
- levenshtein_score = mean(levenshtein)

---

## Captioning CSV

**파일명**: `{timestamp}_{script_name}_captioning.csv`

**태스크**: chebi-20-mol2text, smol-molecule_captioning

| 컬럼 | 타입 | 설명 |
|------|------|------|
| idx | int | 샘플 인덱스 |
| task | str | 태스크 이름 |
| label | str | raw target text |
| pred | str | 생성된 텍스트 |

**오프라인 평가 metric** (corpus 단위 재계산 필요):
- bleu2, bleu4 = nltk.translate.bleu_score.corpus_bleu
- meteor = nltk.translate.meteor_score
- rouge1, rouge2, rougeL = rouge_score

---

## 오프라인 평가 예시

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Classification
df = pd.read_csv("results/20251217/120000_galactica_test_classification.csv")
subset = df[df['task'] == 'bace']
accuracy = subset['correct'].mean()
# 또는 sklearn 사용
# accuracy = accuracy_score(subset['label_parsed'], subset['pred'])

# Regression
df = pd.read_csv("results/20251217/120000_galactica_test_regression.csv")
subset = df[df['task'] == 'qm9_homo']
mae = subset['error'].abs().mean()
rmse = (subset['error'] ** 2).mean() ** 0.5

# Molecule Generation
df = pd.read_csv("results/20251217/120000_galactica_test_molecule_generation.csv")
subset = df[df['task'] == 'retrosynthesis']
validity_ratio = subset['validity'].mean()
exact_match_ratio = subset['exact_match'].mean()
maccs_avg = subset['MACCS_FTS'].mean()

# Captioning (corpus 단위)
df = pd.read_csv("results/20251217/120000_galactica_test_captioning.csv")
subset = df[df['task'] == 'chebi-20-mol2text']
# bleu, rouge는 전체 label/pred 리스트로 재계산
```
