# import json
# import logging
# from pathlib import Path

# import dagshub
# import mlflow
# from mlflow.tracking import MlflowClient


# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------

# MODEL_INFO_PATH = Path("reports/experiment_info.json")
# MODEL_NAME = "model"


# # --------------------------------------------------
# # Logging
# # --------------------------------------------------

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger("model_registry")


# # --------------------------------------------------
# # Utilities
# # --------------------------------------------------

# def load_model_info(path: Path) -> dict:
#     """Load run_id and model_path."""
#     try:
#         with open(path) as f:
#             data = json.load(f)

#         logger.debug("Model info loaded from %s", path)
#         return data

#     except Exception as e:
#         logger.error("Failed loading model info: %s", e)
#         raise


# # --------------------------------------------------
# # Model Registry
# # --------------------------------------------------

# def register_model(model_name: str, model_info: dict):

#     run_id = model_info["run_id"]
#     model_path = model_info["model_path"]

#     model_uri = f"runs:/{run_id}/{model_path}"

#     logger.debug("Registering model from URI: %s", model_uri)

#     model_version = mlflow.register_model(model_uri, model_name)

#     client = MlflowClient()

#     # Set alias (new MLflow registry method)
#     client.set_registered_model_alias(
#         name=model_name,
#         alias="production",
#         version=model_version.version
#     )

#     logger.debug(
#         "Model '%s' version %s registered successfully.",
#         model_name,
#         model_version.version
#     )


# # --------------------------------------------------
# # Main
# # --------------------------------------------------

# def main():

#     logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

#     model_info = load_model_info(MODEL_INFO_PATH)

#     register_model(MODEL_NAME, model_info)


# if __name__ == "__main__":
#     main()