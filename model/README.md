---
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
  - scikit-learn
  - skops
  - tabular-classification
  - gradient-boosting
  - mlops
datasets:
  - motidev/wellness-tourism-dataset
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - roc_auc
---

# Wellness Tourism Purchase Prediction Model

This model estimates whether a customer will purchase a wellness tourism
package. It is the promoted model from an MLOps workflow covering versioned
data, model comparison, MLflow experiment tracking, `skops` serialization,
Hugging Face model storage, and Jenkins-managed application deployment.

Try the deployed application at
[pesapaygateway.com:9450](https://pesapaygateway.com:9450/).

## Model Details

| Property | Value |
|---|---|
| Task | Binary tabular classification |
| Target | `ProdTaken` |
| Positive class | `1` - customer purchased the package |
| Negative class | `0` - customer did not purchase the package |
| Estimator | `GradientBoostingClassifier` |
| Artifact | Complete preprocessing and classifier pipeline |
| Serialization | `skops` |
| Selection metric | F1 score |
| Random state | `42` |

The fitted classifier uses 200 estimators, a learning rate of `0.1`, maximum
tree depth of `5`, and `min_samples_split=2`.

## Intended Use

The model is intended for:

- Demonstrating an end-to-end MLOps workflow for tabular classification.
- Estimating package-purchase propensity from a customer profile and sales
	interaction.
- Supporting prioritization or follow-up decisions when a person reviews the
	prediction alongside business context.

It is not intended to make autonomous eligibility, pricing, credit, health,
employment, or other high-impact decisions. A probability is a model estimate,
not a guarantee that a customer will or will not purchase.

## Training Data

The processed data comes from the
[Wellness Tourism Dataset](https://huggingface.co/datasets/motidev/wellness-tourism-dataset).
Customer identifiers and the source index column were removed before a
stratified 80/20 train/test split.

| Split | Rows | Negative (`0`) | Positive (`1`) |
|---|---:|---:|---:|
| Train | 3,302 | 2,664 | 638 |
| Test | 826 | 667 | 159 |

The positive class represents approximately 19% of each split. F1 was selected
as the primary optimization metric so model comparison would account for both
precision and recall under this class imbalance.

## Preprocessing

The artifact contains preprocessing and prediction in one scikit-learn
`Pipeline`:

- Numerical missing values are imputed with the median.
- Categorical missing values are imputed with the most frequent value.
- Categorical features are one-hot encoded with unknown categories ignored.
- The transformed features are passed to the Gradient Boosting classifier.

Keeping preprocessing in the artifact ensures that training and application
inference use the same transformations.

## Input Schema

The model expects a pandas DataFrame with these 18 columns. Column names are
case-sensitive.

| Feature | Type | Description / observed values |
|---|---|---|
| `Age` | Numeric | Customer age |
| `TypeofContact` | Categorical | `Self Enquiry`, `Company Invited` |
| `CityTier` | Integer | City tier: `1`, `2`, or `3` |
| `DurationOfPitch` | Numeric | Sales pitch duration in minutes |
| `Occupation` | Categorical | `Salaried`, `Small Business`, `Free Lancer`, `Large Business` |
| `Gender` | Categorical | Primarily `Male` or `Female`; source data also contains `Fe Male` |
| `NumberOfPersonVisiting` | Integer | Number of travelers |
| `NumberOfFollowups` | Numeric | Number of sales follow-ups |
| `ProductPitched` | Categorical | `Basic`, `Standard`, `Deluxe`, `Super Deluxe`, `King` |
| `PreferredPropertyStar` | Numeric | Preferred property rating: `3`, `4`, or `5` |
| `MaritalStatus` | Categorical | `Single`, `Married`, `Divorced`, `Unmarried` |
| `NumberOfTrips` | Numeric | Number of trips taken per year |
| `Passport` | Integer | `0` for no, `1` for yes |
| `PitchSatisfactionScore` | Integer | Satisfaction score from `1` to `5` |
| `OwnCar` | Integer | `0` for no, `1` for yes |
| `NumberOfChildrenVisiting` | Numeric | Number of children traveling |
| `Designation` | Categorical | `Executive`, `Manager`, `Senior Manager`, `AVP`, `VP` |
| `MonthlyIncome` | Numeric | Customer monthly income |

## Model Selection

Decision Tree, Random Forest, and Gradient Boosting pipelines were tuned and
compared. Hyperparameter search used five-fold `GridSearchCV` with F1 as the
scoring function. The selected Gradient Boosting configuration achieved a mean
cross-validation F1 of `0.7481` and the strongest held-out F1 among the
candidate pipelines.

Experiments, parameters, metrics, and serialized pipelines were tracked with
MLflow.

## Test Performance

Metrics below were calculated once on the held-out test split.

| Metric | Score |
|---|---:|
| Accuracy | 0.9274 |
| Precision | 0.9160 |
| Recall | 0.6855 |
| F1 | 0.7842 |
| ROC-AUC | 0.9495 |

The default classifier threshold was used. These results describe this test
split and may not transfer unchanged to new populations or later time periods.

## Promotion Policy

Deployment is blocked unless the candidate passes the versioned policy below
on the fixed test split:

| Check | Requirement | Current model |
|---|---:|---:|
| Accuracy | At least 0.90 | 0.9274 |
| Precision | At least 0.85 | 0.9160 |
| Recall | At least 0.65 | 0.6855 |
| F1 | At least 0.75 | 0.7842 |
| ROC-AUC | At least 0.90 | 0.9495 |
| F1 regression | No more than 0.01 below incumbent | Pass |

The gate also requires the expected ordered 18-feature schema and rejects
unexpected serialized types. Jenkins runs the evaluator and its rejection
tests before replacing the deployed container, then archives the JSON decision
report with the build.

## Inference

Install compatible runtime packages:

```bash
pip install pandas scikit-learn skops huggingface_hub
```

Download and load the artifact:

```python
import pandas as pd
import skops.io as sio
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
		repo_id="motidev/wellness-tourism-model",
		filename="wellness_tourism_model.skops",
)

# Review these types before trusting an artifact from an unfamiliar publisher.
trusted_types = sio.get_untrusted_types(file=model_path)
model = sio.load(model_path, trusted=trusted_types)

customer = pd.DataFrame(
		[
				{
						"Age": 35,
						"TypeofContact": "Self Enquiry",
						"CityTier": 1,
						"DurationOfPitch": 15.0,
						"Occupation": "Salaried",
						"Gender": "Male",
						"NumberOfPersonVisiting": 2,
						"NumberOfFollowups": 3.0,
						"ProductPitched": "Basic",
						"PreferredPropertyStar": 3.0,
						"MaritalStatus": "Single",
						"NumberOfTrips": 2.0,
						"Passport": 0,
						"PitchSatisfactionScore": 3,
						"OwnCar": 0,
						"NumberOfChildrenVisiting": 0.0,
						"Designation": "Executive",
						"MonthlyIncome": 25000.0,
				}
		]
)

prediction = int(model.predict(customer)[0])
purchase_probability = float(model.predict_proba(customer)[0, 1])
```

For reproducible production inference, pin `revision` in `hf_hub_download` to
a tested Hugging Face commit instead of following `main`.

## Limitations and Risks

- The dataset is relatively small and the positive class is underrepresented.
- Recall is lower than precision, so the model misses some customers who
	purchase the package.
- The source includes demographic and socioeconomic fields. Performance and
	fairness have not been audited across demographic groups.
- The source contains a gender-label inconsistency (`Female` and `Fe Male`).
- Probability calibration, data drift, and concept drift have not been tested.
- The app accepts a constrained set of values, but the pipeline itself ignores
	previously unseen categories rather than rejecting them.
- Results should be monitored and re-evaluated before use with a different
	customer population or business process.

## Deployment

The Streamlit application downloads this artifact from Hugging Face and serves
predictions from a Docker container. Jenkins builds the image, replaces the
running container, and verifies Streamlit's health endpoint. Hugging Face is
used as the model registry; it is not the application runtime.

Live application: <https://pesapaygateway.com:9450/>
