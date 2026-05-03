import os
from data_science_ml.constants import * 
# from data_science_ml.constants import  ARTIFACT_DIR
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


y = TrainingPipelineConfig().pipeline_name
print(y)