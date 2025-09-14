import mlflow
import dagshub
import json
from pathlib import Path
import logging

# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# initialize dagshub
dagshub.init(repo_owner='VedTholiya', repo_name='swiggy_time_delivery_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VedTholiya/swiggy_time_delivery_prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        return json.load(f)

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"
    
    # Load run information
    run_info = load_model_information(run_info_path)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    logger.info(f"Model artifacts logged with run_id: {run_id}")
    logger.info(f"Model name: {model_name}")
    logger.info("Note: Model registry features are not supported by DagsHub MLflow")