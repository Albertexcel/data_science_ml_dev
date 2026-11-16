import os
import sys

import mlflow
import dagshub
from dotenv import load_dotenv

from data_science_ml.constants import (
    DAGSHUB_REPO_NAME_ENV_KEY,
    DAGSHUB_REPO_OWNER_ENV_KEY,
    DAGSHUB_USER_TOKEN_ENV_KEY,
    MLFLOW_TRACKING_URI_ENV_KEY,
    MLFLOW_LOCAL_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME

)
from data_science_ml.exception import WineException
from data_science_ml.logger import logging


def configure_mlflow_tracking() -> str:
    """
    Point MLflow at the DagsHub tracking server and select the experiment.

    Credentials are read from the environment (a local .env file is loaded first):
      * DAGSHUB_REPO_OWNER + DAGSHUB_REPO_NAME (+ DAGSHUB_USER_TOKEN)  -> dagshub.init(mlflow=True)
      * or MLFLOW_TRACKING_URI (+ MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD)
    If neither is configured the run is tracked in the local ./mlruns folder so
    training still completes.

    :return: the active MLflow tracking URI
    """
    try:
        load_dotenv()

        # MLflow prints emoji in its "View run at: <url>" message, which crashes on Windows' cp1252 console
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

        repo_owner = os.getenv(DAGSHUB_REPO_OWNER_ENV_KEY)
        repo_name = os.getenv(DAGSHUB_REPO_NAME_ENV_KEY)
        user_token = os.getenv(DAGSHUB_USER_TOKEN_ENV_KEY)
        tracking_uri = os.getenv(MLFLOW_TRACKING_URI_ENV_KEY)

        if repo_owner and repo_name:
            import dagshub

            if user_token:
                # Avoids the interactive browser OAuth flow in non-interactive runs
                dagshub.auth.add_app_token(user_token)
            dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
            logging.info(f"DagsHub MLflow tracking enabled for {repo_owner}/{repo_name}")
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logging.info(f"MLflow tracking URI set from environment: {tracking_uri}")
        else:
            local_uri = MLFLOW_LOCAL_TRACKING_URI
            mlflow.set_tracking_uri(local_uri)
            logging.warning(
                "DagsHub is not configured (set DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME / "
                f"DAGSHUB_USER_TOKEN or MLFLOW_TRACKING_URI). Logging to local {MLFLOW_LOCAL_TRACKING_URI} instead."
            )

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        active_uri = mlflow.get_tracking_uri()
        logging.info(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' active at {active_uri}")
        return active_uri
    except Exception as e:
        raise WineException(e, sys) from e
