import os
from data_science_ml.constants import * 
from dataclasses import dataclass
from datetime import datetime

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

"""
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    training_file_path:str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path:str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
"""