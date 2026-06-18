# House Pricing MLOps Pipeline

An end-to-end machine learning pipeline for predicting house sale prices on
the Ames Housing dataset, built with [ZenML](https://zenml.io) for
orchestration, [MLflow](https://mlflow.org) for experiment tracking, and a
[FastAPI](https://fastapi.tiangolo.com) app for real-time inference.

The pipeline covers data ingestion, missing value handling, feature
engineering (column dropping, ordinal encoding, log transforms), outlier
removal, train/test splitting, model training (Linear Regression, Random
Forest, or XGBoost with optional Optuna hyperparameter tuning), and
evaluation. The best trained model is served through a FastAPI app that
loads it directly from ZenML's model registry.

## 1. Setup

### 1.1 Create and activate a virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Get the dataset

Download the [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
(or your source of `archive.zip`) and place it at:

```
data/archive.zip
```

`data_ingestion_step` currently reads from a hardcoded absolute path in
`pipelines/training_pipeline.py`:

```python
raw_data = data_ingestion_step(
    file_path="/Users/m1/Desktop/prices-predictor-system/data/archive.zip"
)
```

**Before running the pipeline, edit this line to point at your own
`archive.zip` location** (e.g. the absolute path to `data/archive.zip` in
your cloned copy of this repo).

### 1.4 Configure the ZenML stack (MLflow tracking + deployment)

This project uses a ZenML stack named `mlflow_stack` with an MLflow
experiment tracker (`mlflow_tracker`). If you haven't set this up yet:

```bash
# Initialize ZenML in this project (if not already done)
zenml init

# Install the MLflow integration
zenml integration install mlflow -y

# Register an MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register a stack combining the default orchestrator/artifact store
# with the MLflow experiment tracker
zenml stack register mlflow_stack \
    -a default \
    -o default \
    -e mlflow_tracker

# Activate the stack
zenml stack set mlflow_stack
```

Verify the active stack:

```bash
zenml stack describe
```

You should see `mlflow_stack` listed with `mlflow_tracker` as its
experiment tracker.

### 1.5 (Optional) Start the ZenML dashboard

```bash
zenml login --local
```

This gives you a local dashboard URL (e.g. `http://127.0.0.1:8237`) to
inspect pipeline runs, steps, and artifacts visually. The training command
below will print a dashboard URL for each run automatically once this is
running.

## 2. Training

Run the training pipeline from the project root:

```bash
python -m pipelines.training_pipeline
```

This runs the full pipeline end to end: ingestion → missing value
handling → feature engineering (ID dropping, ordinal encoding, log
transform) → outlier removal → train/test split → model training →
evaluation. Evaluation metrics (MSE, R²) print at the end of the run.

### 2.1 Choosing a model

The pipeline's `ml_pipeline()` function accepts a `model_type` parameter.
Edit the `if __name__ == "__main__":` block in `pipelines/training_pipeline.py`,
or call it directly in a Python shell:

```python
from pipelines.training_pipeline import ml_pipeline

# Linear Regression (fast, default)
ml_pipeline(model_type="linear_regression")

# Random Forest
ml_pipeline(model_type="random_forest")

# XGBoost
ml_pipeline(model_type="xgboost")
```

### 2.2 Hyperparameter tuning with Optuna

For `random_forest` and `xgboost`, you can enable automatic hyperparameter
tuning. Optuna runs a cross-validated search on the training set only
(the test set is never touched during tuning), then trains a final model
using the best parameters found.

```python
ml_pipeline(
    model_type="xgboost",
    tune_hyperparameters=True,
    n_trials=50,     # number of Optuna trials (default: 30)
    cv_folds=5,      # number of cross-validation folds (default: 5)
)
```

**Note on runtime:** tuning trains `n_trials × cv_folds` models before the
final fit. With `n_trials=50` and `cv_folds=5`, expect this step to take
several minutes (e.g. ~5 minutes for XGBoost on the full Ames dataset).
Start with a smaller `n_trials` (e.g. 15–20) to confirm everything runs
before committing to a long search.

### 2.3 Viewing experiment tracking

Every training run logs parameters, metrics, and the model artifact to
MLflow via ZenML's experiment tracker integration. To browse runs locally:

```bash
mlflow ui --backend-store-uri <path-to-your-mlruns-directory>
```

(Find the correct URI with `zenml experiment-tracker describe mlflow_tracker`.)

### 2.4 Where the trained model goes

Every successful run registers a new version of the `prices_predictor`
model in ZenML's model registry, with the trained `sklearn.pipeline.Pipeline`
stored as an artifact named `sklearn_pipeline`. This is what the deployment
API loads — no manual export step is needed.

## 3. Deployment (real-time inference API)

The `deployment/` app exposes a FastAPI server that loads the latest
`prices_predictor` model from ZenML's registry and serves predictions over
HTTP.

### 3.1 Run the API

From the project root:

```bash
uvicorn deployment.api:app --reload --port 8000
```

On startup, the API loads the latest registered model version and logs
which version it loaded. If no model has been trained yet, the API will
still start, but `/predict` will return a 503 until a model is available.

### 3.2 Interactive docs

Open your browser to:

```
http://127.0.0.1:8000/docs
```

This is FastAPI's auto-generated Swagger UI. It includes a pre-filled
example payload for `/predict` — click **Try it out**, then **Execute**,
to get a prediction without writing any client code.

### 3.3 Endpoints

| Method | Path             | Description                                                        |
|--------|------------------|----------------------------------------------------------------------|
| GET    | `/health`        | Returns API status and the currently loaded model version.         |
| POST   | `/predict`       | Predicts `SalePrice` for a single house given its raw features.    |
| POST   | `/reload-model`  | Force-refreshes the cached model (e.g. after retraining), no restart needed. |

### 3.4 Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MS SubClass": 60, "MS Zoning": "RL", "Lot Frontage": 65.0, "Lot Area": 8450,
    "Street": "Pave", "Alley": null, "Lot Shape": "Reg", "Land Contour": "Lvl",
    "Utilities": "AllPub", "Lot Config": "Inside", "Land Slope": "Gtl",
    "Neighborhood": "CollgCr", "Condition 1": "Norm", "Condition 2": "Norm",
    "Bldg Type": "1Fam", "House Style": "2Story", "Overall Qual": 7, "Overall Cond": 5,
    "Year Built": 2003, "Year Remod/Add": 2003, "Roof Style": "Gable", "Roof Matl": "CompShg",
    "Exterior 1st": "VinylSd", "Exterior 2nd": "VinylSd", "Mas Vnr Type": "BrkFace",
    "Mas Vnr Area": 196.0, "Exter Qual": "Gd", "Exter Cond": "TA", "Foundation": "PConc",
    "Bsmt Qual": "Gd", "Bsmt Cond": "TA", "Bsmt Exposure": "No", "BsmtFin Type 1": "GLQ",
    "BsmtFin SF 1": 706.0, "BsmtFin Type 2": "Unf", "BsmtFin SF 2": 0.0, "Bsmt Unf SF": 150.0,
    "Total Bsmt SF": 856.0, "Bsmt Full Bath": 1, "Bsmt Half Bath": 0, "Heating": "GasA",
    "Heating QC": "Ex", "Central Air": "Y", "Electrical": "SBrkr", "1st Flr SF": 856.0,
    "2nd Flr SF": 854.0, "Low Qual Fin SF": 0.0, "Gr Liv Area": 1710.0, "Full Bath": 2,
    "Half Bath": 1, "Bedroom AbvGr": 3, "Kitchen AbvGr": 1, "Kitchen Qual": "Gd",
    "TotRms AbvGrd": 8, "Functional": "Typ", "Fireplaces": 1, "Fireplace Qu": "TA",
    "Garage Type": "Attchd", "Garage Yr Blt": 2003.0, "Garage Finish": "RFn",
    "Garage Cars": 2.0, "Garage Area": 548.0, "Garage Qual": "TA", "Garage Cond": "TA",
    "Paved Drive": "Y", "Wood Deck SF": 0.0, "Open Porch SF": 61.0, "Enclosed Porch": 0.0,
    "3Ssn Porch": 0.0, "Screen Porch": 0.0, "Pool Area": 0.0, "Pool QC": null, "Fence": null,
    "Misc Feature": null, "Misc Val": 0.0, "Mo Sold": 2, "Yr Sold": 2008, "Sale Type": "WD ",
    "Sale Condition": "Normal"
  }'
```

Expected response:

```json
{
  "predicted_sale_price": 208500.0,
  "log_predicted_sale_price": 12.247,
  "model_version": "15"
}
```

### 3.5 Field validation rules

Most fields are required — a missing value there usually means bad input
data. A specific set of categorical fields, however, accept `null`,
because in the Ames Housing dataset a missing value there means **"this
feature doesn't apply to this house"** rather than missing data:

| Field(s)                                                                 | `null` means              |
|---------------------------------------------------------------------------|----------------------------|
| `Alley`                                                                    | No alley access            |
| `Mas Vnr Type`                                                             | No masonry veneer           |
| `Bsmt Qual`, `Bsmt Cond`, `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2` | No basement                |
| `Fireplace Qu`                                                             | No fireplace                |
| `Garage Type`, `Garage Yr Blt`, `Garage Finish`, `Garage Qual`, `Garage Cond` | No garage                   |
| `Pool QC`                                                                  | No pool                     |
| `Fence`                                                                    | No fence                    |
| `Misc Feature`                                                             | No additional feature       |

Sending `null` for any other field will return a `422 Unprocessable
Entity` validation error.

## 4. Retraining and redeploying

After retraining (step 2), the new model is automatically registered as a
new version of `prices_predictor`. To make the running API pick it up
without restarting it:

```bash
curl -X POST "http://127.0.0.1:8000/reload-model"
```
