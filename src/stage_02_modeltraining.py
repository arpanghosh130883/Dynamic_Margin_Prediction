import argparse
import os
#import shutil
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import joblib

STAGE = "Model_Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


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


    ccp_alpha = params["train"]["ccp_alpha"]
    max_depth = params["train"]["max_depth"]
    min_samples_leaf = params["train"]["min_samples_leaf"]
    min_samples_split = params["train"]["min_samples_split"]
    n_estimators = params["train"]["n_estimators"]

    mlflow.log_params(params["train"])

    #Model creation
    model = RandomForestRegressor(ccp_alpha=ccp_alpha, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                      n_estimators=n_estimators)
    ## Tuned RFRegressor with post prunning
    model.fit(train_x,train_y)

    #Creating model directory
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    create_directories([model_dir_path])

    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    #Dumping the model in Model directory folder
    joblib.dump(model, model_path)
    #mlflow.sklearn.log_model(model, "RFregressorBST", registered_model_name="RFregressorBST")

    #Mlflow logging of model
    mlflow.sklearn.log_model(model, "RFregressorBST")



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