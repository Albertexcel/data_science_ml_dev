import os
import shutil
import sys
import threading
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from data_science_ml.constants import (
    MODEL_PREDICTION_CONFIDENCE_COLUMN_NAME,
    MODEL_PREDICTION_ERROR_COLUMN_NAME,
    MODEL_PREDICTION_LABEL_COLUMN_NAME,
    MODEL_PREDICTION_PROBABILITY_COLUMN_PREFIX,
    MODEL_PREDICTION_STATUS_COLUMN_NAME,
    PREPROCESSING_OBJECT_FILE_NAME,
    TARGET_COLUMN,
)
from data_science_ml.entity.artifact_entity import (
    ModelPredictionArtifact,
    PredictionInputValidationArtifact,
    PredictionModelArtifact,
)
from data_science_ml.entity.config_entity import ModelPredictionConfig
from data_science_ml.entity.estimator import TargetValueMapping, WineModel
from data_science_ml.exception import WineException
from data_science_ml.logger import logging
from data_science_ml.utils.common_functions import get_schema_column_names, load_object, read_yaml_file, read_tabular_file

CACHE_COMPLETE_MARKER = "_COMPLETE"


class ModelPrediction:
    """
    Serves the production version of the registered model on DagsHub (MLflow model registry).

    The registered MLflow model only contains the bare sklearn estimator (see ModelTrainer._log_to_mlflow),
    so the fitted preprocessor is fetched from `<run>/preprocessing/preprocessing.pkl` of the run that
    produced the registered version and both are wrapped in WineModel. Pairing by run id guarantees the
    exact training-time scaling is used.

    Downloads are cached on disk per model version and loaded models are cached in memory per process,
    so repeated predictions never re-download or re-load the model. Nothing here ever calls fit().
    """
    _loaded_models: Dict[Tuple[str, str, str], Tuple[WineModel, PredictionModelArtifact]] = {}
    _lock = threading.Lock()

    def __init__(self, model_prediction_config: ModelPredictionConfig):
        """
        :param model_prediction_config: configuration for model prediction (model name, alias/version, cache paths)
        """
        try:
            self.model_prediction_config = model_prediction_config
            self._schema_config = read_yaml_file(file_path=self.model_prediction_config.schema_file_path)
            self.model: WineModel = None
            self.model_artifact: PredictionModelArtifact = None
        except Exception as e:
            raise WineException(e, sys) from e

    def _get_cache_dir(self, version: str) -> str:
        return os.path.join(self.model_prediction_config.model_cache_dir,
                            self.model_prediction_config.registered_model_name, f"v{version}")

    def _read_cached_run_id(self, version: str):
        marker = os.path.join(self._get_cache_dir(version), CACHE_COMPLETE_MARKER)
        if os.path.exists(marker):
            with open(marker) as f:
                return f.read().strip() or None
        return None

    def get_model_version(self, client) -> Tuple[object, str]:
        """
        Method Name :   get_model_version
        Description :   Resolves the registered model version to serve. Order: pinned version -> alias -> stage

        Output      :   Returns (ModelVersion, description of how it was resolved)
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            config = self.model_prediction_config
            name = config.registered_model_name
            if config.model_version:
                return client.get_model_version(name, str(config.model_version)), \
                    f"pinned version {config.model_version}"

            errors = []
            if config.model_alias:
                try:
                    return client.get_model_version_by_alias(name, config.model_alias), f"alias @{config.model_alias}"
                except Exception as e:
                    errors.append(f"alias '@{config.model_alias}' not found ({type(e).__name__})")

            if config.model_stage:
                stage = config.model_stage.lower()
                versions = [v for v in client.search_model_versions(f"name='{name}'")
                            if (v.current_stage or "").lower() == stage]
                if versions:
                    return max(versions, key=lambda v: int(v.version)), f"stage {config.model_stage}"
                errors.append(f"no version in stage '{config.model_stage}'")

            raise ValueError(
                f"No production version found for registered model '{name}': {'; '.join(errors)}. "
                f"Assign the alias '{config.model_alias}' to a version on DagsHub, "
                f"or pin one with the MLFLOW_MODEL_VERSION environment variable."
            )
        except Exception as e:
            raise WineException(e, sys) from e

    def download_model_artifacts(self, version: str, run_id: str) -> Tuple[str, str]:
        """
        Method Name :   download_model_artifacts
        Description :   Downloads the registered model and the run's preprocessing object into the local cache.
                        Writes into a temp dir first so an interrupted download never leaves a half-filled cache.

        Output      :   Returns (local model dir, local preprocessing object file path)
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            import mlflow

            config = self.model_prediction_config
            cache_dir = self._get_cache_dir(version)
            model_dir = os.path.join(cache_dir, "model")
            preprocessing_path = os.path.join(cache_dir, PREPROCESSING_OBJECT_FILE_NAME)

            if self._read_cached_run_id(version) == run_id:
                logging.info(f"Using cached model artifacts at {cache_dir}")
                return model_dir, preprocessing_path

            tmp_dir = f"{cache_dir}.downloading"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir, exist_ok=True)

            model_uri = f"models:/{config.registered_model_name}/{version}"
            logging.info(f"Downloading {model_uri} to {tmp_dir}")
            downloaded_model_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=os.path.join(tmp_dir, "model"))

            logging.info(f"Downloading {config.preprocessing_artifact_path} from run {run_id}")
            try:
                downloaded_preprocessing = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=config.preprocessing_artifact_path, dst_path=tmp_dir)
            except Exception as e:
                raise FileNotFoundError(
                    f"Run {run_id} (model version {version}) has no '{config.preprocessing_artifact_path}' "
                    f"artifact, so its training-time preprocessing cannot be reproduced: {e}"
                ) from e

            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            shutil.move(downloaded_model_dir, model_dir)
            shutil.move(downloaded_preprocessing, preprocessing_path)
            with open(os.path.join(cache_dir, CACHE_COMPLETE_MARKER), "w") as f:
                f.write(run_id)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logging.info(f"Cached model version {version} at {cache_dir}")
            return model_dir, preprocessing_path

        except Exception as e:
            raise WineException(e, sys) from e

    @staticmethod
    def get_feature_ranges(preprocessor) -> Dict[str, Tuple[float, float]]:
        """Training min/max of each feature, taken from the fitted MinMaxScaler."""
        ranges = {}
        for _, transformer, columns in getattr(preprocessor, "transformers_", []):
            if hasattr(transformer, "data_min_") and isinstance(columns, (list, tuple, np.ndarray)):
                for col, low, high in zip(columns, transformer.data_min_, transformer.data_max_):
                    ranges[str(col)] = (float(low), float(high))
        return ranges

    def get_model_artifact(self, preprocessor, trained_model, version: str, run_id: str, resolved_by: str,
                           model_dir: str, preprocessing_path: str) -> PredictionModelArtifact:
        """
        Method Name :   get_model_artifact
        Description :   Checks that the preprocessor and the model belong together and describes what the model expects

        Output      :   Returns PredictionModelArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            schema_features = get_schema_column_names(self._schema_config, "numerical_columns")
            expected_features = [str(c) for c in getattr(preprocessor, "feature_names_in_", schema_features)]

            n_transformed = len(preprocessor.get_feature_names_out())
            n_model = getattr(trained_model, "n_features_in_", n_transformed)
            if n_transformed != n_model:
                raise ValueError(
                    f"Preprocessor outputs {n_transformed} features but model version {version} expects {n_model}. "
                    "The preprocessing artifact does not match the registered model."
                )

            reverse_mapping = TargetValueMapping().reverse_mapping()
            classes = list(getattr(trained_model, "classes_", []))
            unknown = [c for c in classes if int(c) not in reverse_mapping]
            if unknown:
                raise ValueError(f"Model classes {unknown} are not in TargetValueMapping {reverse_mapping}")

            registered_model_name = self.model_prediction_config.registered_model_name
            return PredictionModelArtifact(
                registered_model_name=registered_model_name,
                model_version=str(version),
                resolved_by=resolved_by,
                run_id=run_id,
                model_uri=f"models:/{registered_model_name}/{version}",
                local_model_dir=model_dir,
                preprocessing_object_file_path=preprocessing_path,
                expected_features=expected_features,
                class_labels=[reverse_mapping[int(c)] for c in classes],
                feature_ranges=self.get_feature_ranges(preprocessor),
            )
        except Exception as e:
            raise WineException(e, sys) from e

    def load_model(self, force_reload: bool = False) -> PredictionModelArtifact:
        """
        Method Name :   load_model
        Description :   Resolves, downloads (once) and loads the production model with its preprocessor

        Output      :   Returns PredictionModelArtifact, and sets self.model / self.model_artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered load_model method of ModelPrediction class")
        try:
            if self.model is not None and not force_reload:
                return self.model_artifact

            import mlflow.sklearn

            config = self.model_prediction_config
            with self._lock:
                pinned = config.model_version
                cached_run_id = self._read_cached_run_id(pinned) if pinned else None

                if pinned and cached_run_id and not force_reload:
                    # A registry version is immutable, so a pinned and fully cached version needs no network call
                    version, run_id, resolved_by = str(pinned), cached_run_id, f"pinned version {pinned} (cached)"
                else:
                    from mlflow import MlflowClient
                    from data_science_ml.configuration.mlflow_connection import configure_mlflow_tracking

                    tracking_uri = configure_mlflow_tracking()
                    logging.info(f"Resolving production model from {tracking_uri}")
                    model_version, resolved_by = self.get_model_version(MlflowClient())
                    version, run_id = str(model_version.version), model_version.run_id
                    if not run_id:
                        raise ValueError(f"Model version {version} has no source run, "
                                         "so its preprocessing object cannot be located.")

                key = (config.model_cache_dir, config.registered_model_name, version)
                if key in self._loaded_models and not force_reload:
                    logging.info(f"Model version {version} already loaded in memory")
                    self.model, self.model_artifact = self._loaded_models[key]
                    return self.model_artifact

                if force_reload:
                    shutil.rmtree(self._get_cache_dir(version), ignore_errors=True)

                model_dir, preprocessing_path = self.download_model_artifacts(version, run_id)
                trained_model = mlflow.sklearn.load_model(model_dir)
                preprocessor = load_object(file_path=preprocessing_path)

                model_artifact = self.get_model_artifact(preprocessor, trained_model, version, run_id,
                                                         resolved_by, model_dir, preprocessing_path)
                wine_model = WineModel(preprocessing_object=preprocessor, trained_model_object=trained_model)
                self._loaded_models[key] = (wine_model, model_artifact)
                self.model, self.model_artifact = wine_model, model_artifact
                logging.info(f"Loaded {wine_model} from {model_artifact.model_uri} ({resolved_by}, run {run_id})")
                return model_artifact

        except Exception as e:
            raise WineException(e, sys) from e

    @staticmethod
    def _normalise_column_name(column) -> str:
        return str(column).strip().lower().replace(" ", "_").replace("-", "_")

    def validate_input(self, dataframe: DataFrame) -> Tuple[DataFrame, PredictionInputValidationArtifact]:
        """
        Method Name :   validate_input
        Description :   Checks the raw input against the features the production model was trained on,
                        coerces them to numeric and flags rows that cannot be scored

        Output      :   Returns (feature DataFrame in training column order, PredictionInputValidationArtifact)
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered validate_input method of ModelPrediction class")
        try:
            self.load_model()
            expected_features = self.model_artifact.expected_features

            if not isinstance(dataframe, DataFrame) or dataframe.empty:
                return DataFrame(), PredictionInputValidationArtifact(
                    validation_status=False, message="The input dataset is empty.")

            if len(dataframe) > self.model_prediction_config.max_rows:
                return DataFrame(), PredictionInputValidationArtifact(
                    validation_status=False,
                    message=f"The input has {len(dataframe)} rows; the limit is "
                            f"{self.model_prediction_config.max_rows}. Split the file into smaller batches.")

            df = dataframe.reset_index(drop=True)

            # Accept harmless header variations such as "Fixed Acidity" or "fixed-acidity"
            rename_map = {}
            for col in df.columns:
                normalised = self._normalise_column_name(col)
                if col not in expected_features and normalised in expected_features:
                    rename_map[col] = normalised
            df = df.rename(columns=rename_map)

            duplicated = sorted({str(c) for c in df.columns[df.columns.duplicated()] if c in expected_features})
            if duplicated:
                return DataFrame(), PredictionInputValidationArtifact(
                    validation_status=False, message=f"Duplicate feature columns found: {duplicated}.")

            missing_columns = [c for c in expected_features if c not in df.columns]
            ignored_columns = [str(c) for c in df.columns if c not in expected_features]
            if missing_columns:
                message = (f"Missing required columns: {missing_columns}. "
                           f"The model expects: {expected_features}.")
                logging.info(message)
                return DataFrame(), PredictionInputValidationArtifact(
                    validation_status=False, message=message,
                    missing_columns=missing_columns, ignored_columns=ignored_columns)

            # Only the training features are passed on; the target and any extra columns never reach the model
            raw_features = df[expected_features]
            features = raw_features.apply(pd.to_numeric, errors="coerce").astype(float)
            features = features.replace([np.inf, -np.inf], np.nan)

            row_errors: Dict[int, list] = {}
            for col in expected_features:
                for idx in features.index[features[col].isna()]:
                    raw_value = raw_features.at[idx, col]
                    reason = "missing value" if pd.isna(raw_value) else f"'{raw_value}' is not a valid number"
                    row_errors.setdefault(int(idx), []).append(f"{col}: {reason}")
            invalid_rows = {idx: "; ".join(errors) for idx, errors in row_errors.items()}

            out_of_range_columns = {}
            valid_features = features.drop(index=list(invalid_rows))
            for col, (low, high) in self.model_artifact.feature_ranges.items():
                if col in valid_features:
                    n_outside = int(((valid_features[col] < low) | (valid_features[col] > high)).sum())
                    if n_outside:
                        out_of_range_columns[col] = n_outside

            n_valid = len(df) - len(invalid_rows)
            validation_status = n_valid > 0
            message = f"{n_valid} of {len(df)} rows are valid for prediction."
            if invalid_rows:
                message += f" {len(invalid_rows)} rows have missing or non-numeric feature values and will be skipped."
            if TARGET_COLUMN in ignored_columns:
                message += f" Column '{TARGET_COLUMN}' is the training target and is ignored."
            if out_of_range_columns:
                message += (f" Some values fall outside the training range (predictions may be less reliable): "
                            f"{out_of_range_columns}.")

            validation_artifact = PredictionInputValidationArtifact(
                validation_status=validation_status,
                message=message,
                missing_columns=[],
                ignored_columns=ignored_columns,
                invalid_rows=invalid_rows,
                out_of_range_columns=out_of_range_columns,
            )
            logging.info(f"Prediction input validation: {message}")
            return features, validation_artifact

        except Exception as e:
            raise WineException(e, sys) from e

    def get_prediction_dataframe(self, dataframe: DataFrame, features: DataFrame,
                                 validation_artifact: PredictionInputValidationArtifact) -> DataFrame:
        """
        Method Name :   get_prediction_dataframe
        Description :   Scores the valid rows through WineModel (training-time preprocessing + model) and appends
                        label, confidence, per-class probabilities, status and validation error to the input

        Output      :   Returns the input DataFrame with the prediction columns appended
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            output_df = dataframe.reset_index(drop=True).copy()
            n_rows = len(output_df)
            valid_mask = ~features.index.isin(list(validation_artifact.invalid_rows))
            valid_features = features[valid_mask]

            reverse_mapping = TargetValueMapping().reverse_mapping()
            predictions = self.model.predict(valid_features)
            probabilities = self.model.predict_proba(valid_features)

            labels = np.full(n_rows, None, dtype=object)
            labels[valid_mask] = [reverse_mapping[int(p)] for p in predictions]
            output_df[MODEL_PREDICTION_LABEL_COLUMN_NAME] = labels

            if probabilities is not None:
                full_probabilities = np.full((n_rows, probabilities.shape[1]), np.nan)
                full_probabilities[valid_mask] = probabilities
                confidence = np.full(n_rows, np.nan)
                confidence[valid_mask] = probabilities.max(axis=1)
                output_df[MODEL_PREDICTION_CONFIDENCE_COLUMN_NAME] = confidence
                for i, label in enumerate(self.model_artifact.class_labels):
                    output_df[f"{MODEL_PREDICTION_PROBABILITY_COLUMN_PREFIX}{label}"] = full_probabilities[:, i]

            output_df[MODEL_PREDICTION_STATUS_COLUMN_NAME] = np.where(valid_mask, "success", "failed")
            output_df[MODEL_PREDICTION_ERROR_COLUMN_NAME] = [validation_artifact.invalid_rows.get(i, "")
                                                             for i in range(n_rows)]
            return output_df

        except Exception as e:
            raise WineException(e, sys) from e

    def save_predictions(self, prediction_df: DataFrame) -> str:
        """Writes predictions to <prediction_output_dir>/<timestamp>/predictions.csv and returns the path."""
        try:
            output_dir = os.path.join(self.model_prediction_config.prediction_output_dir,
                                      datetime.now().strftime('%m_%d_%Y_%H_%M_%S'))
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, self.model_prediction_config.prediction_output_file_name)
            prediction_df.to_csv(file_path, index=False)
            logging.info(f"Saved predictions to {file_path}")
            return file_path
        except Exception as e:
            raise WineException(e, sys) from e

    def initiate_model_prediction(self, dataframe: DataFrame, save_output: bool = False) -> ModelPredictionArtifact:
        """
        Method Name :   initiate_model_prediction
        Description :   This function initiates the model prediction steps: load the production model,
                        validate the raw input and score it. Rows that fail validation are kept in the
                        output with a failed status.

        Output      :   Returns ModelPredictionArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_prediction method of ModelPrediction class")
        try:
            self.load_model()
            features, validation_artifact = self.validate_input(dataframe)
            if not validation_artifact.validation_status:
                raise ValueError(validation_artifact.message)

            prediction_df = self.get_prediction_dataframe(dataframe, features, validation_artifact)
            n_failed = len(validation_artifact.invalid_rows)

            model_prediction_artifact = ModelPredictionArtifact(
                prediction_dataframe=prediction_df,
                model_artifact=self.model_artifact,
                validation_artifact=validation_artifact,
                total_records=len(prediction_df),
                predicted_records=len(prediction_df) - n_failed,
                failed_records=n_failed,
            )
            if save_output:
                model_prediction_artifact.prediction_file_path = self.save_predictions(prediction_df)

            logging.info(f"Predicted {model_prediction_artifact.predicted_records}/{len(prediction_df)} rows with "
                         f"{self.model_artifact.model_uri}")
            return model_prediction_artifact

        except Exception as e:
            raise WineException(e, sys) from e


if __name__ == "__main__":

    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "wine_quality_classification.csv")
    model_prediction_artifact = ModelPrediction(
        model_prediction_config=ModelPredictionConfig(),
    ).initiate_model_prediction(read_tabular_file(input_path), save_output=True)
    print(model_prediction_artifact.validation_artifact.message)
    print(model_prediction_artifact.prediction_dataframe.head())
