from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pandas import DataFrame


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class PredictionModelArtifact:
    registered_model_name: str
    model_version: str
    resolved_by: str
    run_id: str
    model_uri: str
    local_model_dir: str
    preprocessing_object_file_path: str
    expected_features: List[str]
    class_labels: List[str]
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

@dataclass
class PredictionInputValidationArtifact:
    validation_status: bool
    message: str
    missing_columns: List[str] = field(default_factory=list)
    ignored_columns: List[str] = field(default_factory=list)
    invalid_rows: Dict[int, str] = field(default_factory=dict)
    out_of_range_columns: Dict[str, int] = field(default_factory=dict)

@dataclass
class ModelPredictionArtifact:
    prediction_dataframe: DataFrame
    model_artifact: PredictionModelArtifact
    validation_artifact: PredictionInputValidationArtifact
    total_records: int
    predicted_records: int
    failed_records: int

