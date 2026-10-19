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
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

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

