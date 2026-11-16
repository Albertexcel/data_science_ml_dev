import os
from data_science_ml.constants import * 
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from data_science_ml.utils.common_functions import get_env_variable

TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = "data_science_ml"
    artifact_dir: str = os.path.join("artifact", TIMESTAMP)
    timestamp:str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
    training_file_path:str = os.path.join(data_ingestion_dir, "data_ingested", "train.csv")
    testing_file_path:str = os.path.join(data_ingestion_dir, "data_ingested", "test.csv")



@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_f1_score: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

@dataclass
class ModelPredictionConfig:
    """
    Which registered model to serve and where to cache it. Resolution order for the model version:
    MLFLOW_MODEL_VERSION (explicit pin) -> MLFLOW_MODEL_ALIAS (default "production")
    -> MLFLOW_MODEL_STAGE (legacy stage, default "Production"). Nothing silently falls back to "latest".
    Lives under artifact/prediction rather than a timestamped run folder because the cached model is
    reused across many prediction requests. Values are read when the config is created, not at import.
    """
    registered_model_name: str = field(default_factory=lambda: get_env_variable(
        "MLFLOW_REGISTERED_MODEL_NAME", MLFLOW_REGISTERED_MODEL_NAME))
    model_version: Optional[str] = field(default_factory=lambda: get_env_variable("MLFLOW_MODEL_VERSION"))
    model_alias: Optional[str] = field(default_factory=lambda: get_env_variable(
        "MLFLOW_MODEL_ALIAS", MODEL_PREDICTION_MODEL_ALIAS))
    model_stage: Optional[str] = field(default_factory=lambda: get_env_variable(
        "MLFLOW_MODEL_STAGE", MODEL_PREDICTION_MODEL_STAGE))
    preprocessing_artifact_path: str = f"{MLFLOW_PREPROCESSING_ARTIFACT_DIR}/{PREPROCESSING_OBJECT_FILE_NAME}"
    model_cache_dir: str = field(default_factory=lambda: get_env_variable(
        "PREDICTION_MODEL_CACHE_DIR",
        os.path.join(ARTIFACT_DIR, MODEL_PREDICTION_DIR_NAME, MODEL_PREDICTION_MODEL_CACHE_DIR)))
    prediction_output_dir: str = field(default_factory=lambda: get_env_variable(
        "PREDICTION_OUTPUT_DIR",
        os.path.join(ARTIFACT_DIR, MODEL_PREDICTION_DIR_NAME, MODEL_PREDICTION_OUTPUT_DIR)))
    prediction_output_file_name: str = MODEL_PREDICTION_OUTPUT_FILE_NAME
    schema_file_path: str = SCHEMA_FILE_PATH
    max_rows: int = MODEL_PREDICTION_MAX_ROWS
