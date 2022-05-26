import argparse
import os
import numpy as np
#from tqdm import tqdm
import logging

from sklearn.metrics import r2_score
from src.utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import joblib

STAGE = "Model_Evaluation" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def evaluate(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main(config_path, params_path):

    #mlflow.set_tracking_uri("http://127.0.0.1:1234")  # for storing in SQllite mlflow.db

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)


    #Reading the Train test data from the respective folder
    artifacts = config["artifacts"]
    training_test_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["TRAIN_TEST_DATA"])
    train_data_path = os.path.join(training_test_data_dir_path, artifacts["TRAIN_DATA"])
    df_train = pd.read_csv(train_data_path, sep=",")
    test_data_path = os.path.join(training_test_data_dir_path, artifacts["TEST_DATA"])
    df_test = pd.read_csv(test_data_path, sep=",")

    
    train_y = df_train[["Margin"]]
    test_y = df_test[["Margin"]]
    train_x = df_train.drop(['Margin'], axis=1)
    test_x = df_test.drop(['Margin'], axis=1)


    #Reading the Model pickle file
    artifacts = config["artifacts"]
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    model = joblib.load(model_path)

    pred = model.predict(test_x)

    rmse, mae, r2 = evaluate(test_y, pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

