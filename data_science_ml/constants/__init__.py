import os

DATABASE = "ML;"
TABLE = "wine_quality_classification"
ARTIFACT_DIR = "artifact"
MODEL_FILE_NAME = "model.pkl"
PIPELINE_NAME = "data_science_ml"

TARGET_COLUMN = "quality_label"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "schema.yaml")
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Ingestion related constants
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR = "data_ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("data_science_ml", "config", "model.yaml")


"""
MLflow related constant
"""
MLFLOW_MODEL_ARTIFACT_NAME: str = "model"
MLFLOW_EXPERIMENT_NAME: str = "wine_classification"
MLFLOW_PREPROCESSING_ARTIFACT_DIR: str = "preprocessing"
MLFLOW_REGISTERED_MODEL_NAME: str = "wine_model"

# Names of the environment variables (values live in .env, never in code)
DAGSHUB_REPO_NAME_ENV_KEY: str = "DAGSHUB_REPO_NAME"
DAGSHUB_REPO_OWNER_ENV_KEY: str = "DAGSHUB_REPO_OWNER"
DAGSHUB_USER_TOKEN_ENV_KEY: str = "DAGSHUB_USER_TOKEN"
MLFLOW_TRACKING_URI_ENV_KEY: str = "MLFLOW_TRACKING_URI"
MLFLOW_LOCAL_TRACKING_URI: str = "sqlite:///mlflow.db"


"""
MODEL PREDICTION related constant start with MODEL_PREDICTION var name
The model name / alias / stage below are defaults only; they can be overridden with environment variables
(or in wine_classification/.env) so model versions are never hardcoded at deploy time.
"""
MODEL_PREDICTION_DIR_NAME: str = "prediction"
MODEL_PREDICTION_MODEL_CACHE_DIR: str = "model_cache"
MODEL_PREDICTION_OUTPUT_DIR: str = "outputs"
MODEL_PREDICTION_OUTPUT_FILE_NAME: str = "predictions.csv"
MODEL_PREDICTION_MODEL_ALIAS: str = "production"
MODEL_PREDICTION_MODEL_STAGE: str = "Production"
MODEL_PREDICTION_MAX_ROWS: int = 100_000
MODEL_PREDICTION_SUPPORTED_FILE_EXTENSIONS: tuple = (".csv", ".xlsx")
MODEL_PREDICTION_LABEL_COLUMN_NAME: str = "predicted_quality_label"
MODEL_PREDICTION_CONFIDENCE_COLUMN_NAME: str = "prediction_confidence"
MODEL_PREDICTION_PROBABILITY_COLUMN_PREFIX: str = "probability_"
MODEL_PREDICTION_STATUS_COLUMN_NAME: str = "prediction_status"
MODEL_PREDICTION_ERROR_COLUMN_NAME: str = "validation_error"
