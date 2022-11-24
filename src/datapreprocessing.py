import argparse
import os
#import shutil
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml
import pandas as pd
#from sklearn.model_selection import train_test_split
#import mlflow
#import random
import pickle


STAGE = "Data_Preprocessing" ## <<< change stage name 

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

    #Reading the data from the source folder
    source_data = config["source_data"]
    input_data = os.path.join(source_data["data_dir"], source_data["data_file"])

    #Converting the data into dataframe
    df = pd.read_csv(input_data, sep=",")

    #Reading the Pickel file for Categorical to Numerical Encoding
    artifacts = config["artifacts"]
    pickle_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PICKLE_DIR"])
    aza_pickle_path = os.path.join(pickle_dir_path, artifacts["AZA_PICKLE"])
    source_pickle_path = os.path.join(pickle_dir_path, artifacts["SOURCE_PICKLE"])
    channel_pickle_path = os.path.join(pickle_dir_path, artifacts["CHANNEL_PICKLE"])
    scaler_pickle_path = os.path.join(pickle_dir_path, artifacts["SCALER_PICKLE"])
    
    #Reading the Model Pickel file 
    artifacts = config["artifacts"]
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])


    # Importing the Pickel files from artifacts directory
    with open(aza_pickle_path, 'rb') as f:
        ordered_labels_aza_TB = pickle.load(f)
    with open(channel_pickle_path, 'rb') as f:
        ordinal_label_channel_TB = pickle.load(f)
    with open(source_pickle_path, 'rb') as f:
        ordinal_label_source_TB = pickle.load(f)
    with open(scaler_pickle_path, 'rb') as f:
        scaler_WTB = pickle.load(f)
    with open(model_path, 'rb') as f:
        RFregressorBSTRS_WTB = pickle.load(f)

    #NULL Treatment
    df['aza'].fillna(params["fillnull"]["aza"], inplace=True)
    df['source'].fillna(params["fillnull"]["source"], inplace=True)
    df['channel'].fillna(params["fillnull"]["channel"], inplace=True)
    df['Age'].fillna(params["fillnull"]["age"], inplace=True)

    #Categorical to Numeric Encoding
    df.channel = df.channel.map(ordinal_label_channel_TB)
    df.source = df.source.map(ordinal_label_source_TB)
    df.aza = df.aza.map(ordered_labels_aza_TB)

    #If NULL for any new parameters not present in Training data
    df['aza'].fillna(params["fillnull"]["London"], inplace=True)
    df['source'].fillna(params["fillnull"]["Affiliate"], inplace=True)
    df['channel'].fillna(params["fillnull"]["Website"], inplace=True)

    #df['aza'].fillna(1.991306, inplace=True)  # London
    #df['source'].fillna(15, inplace=True)  # Affiliate
    #df['channel'].fillna(4, inplace=True)  # Website
    
    #Normalization
    df_scaled = scaler_WTB.transform(df)

    #Prediction
    yhat_WTB = RFregressorBSTRS_WTB.predict(df_scaled)



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