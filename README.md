# Bank Account Prediction - Financial Inclusion in Africa

Binary classifier predicting whether a survey respondent has a bank account.

## Problem description

This project predicts bank-account ownership in East Africa to support financial-inclusion efforts. Using 23,524 survey responses from Kenya, Rwanda, Tanzania and Uganda, it models whether a respondent has a bank account from demographic and socioeconomic features such as age, household size, phone access, education, marital status and occupation.

The target `bank_account` is imbalanced (about 14% "Yes"), so the work uses a stratified train/validation split and evaluates with ROC-AUC and F1 alongside accuracy. A scikit-learn `ColumnTransformer` scales numeric features and one-hot encodes categoricals; logistic regression, random forest and XGBoost are compared, then XGBoost is tuned with 5-fold grid search. The full fitted pipeline is saved as `model.bin`.

A Flask API serves predictions at `/predict`, returning a probability and a binary decision. The service is containerised with Docker and deployed on Render. Intended users are banks, fintechs, NGOs and policymakers who need to identify likely unbanked groups and target outreach or product design.

## Setup

On macOS, install the OpenMP library required by XGBoost:

```bash
brew install libomp
```

Create and activate a virtual environment, then install dependencies:

```bash
cd "/Users/aguuchechukwueric/Desktop/ML MID TERM PROJECT/bank"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that XGBoost imports:

```bash
python -c "import xgboost; print('xgboost ok', xgboost.__version__)"
```

## Data

Place the training CSV either at `data/Train.csv` or use the existing `rawdata.csv`
in the project root. `train.py` looks in both locations automatically.

## Train

```bash
python train.py
```

This prints metrics for logistic regression, random forest, XGBoost and a tuned
XGBoost pipeline, then saves the full pipeline to `model.bin`.

## Predict

Start the Flask service:

```bash
python predict.py
```

In another terminal:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"Kenya","year":2018,"location_type":"Rural","cellphone_access":"Yes","household_size":3,"age_of_respondent":24,"gender_of_respondent":"Female","relationship_with_head":"Spouse","marital_status":"Married/Living together","education_level":"Secondary education","job_type":"Self employed"}'
```

## Docker

Build and run the container:

```bash
docker build -t financial-inclusion .
docker run -p 9696:9696 financial-inclusion
```

Then repeat the curl request against `http://localhost:9696/predict`.
