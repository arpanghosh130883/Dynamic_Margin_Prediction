import argparse
import os
#import shutil
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import random


STAGE = "Data_Ingestion" ## <<< change stage name 

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

    #### Count frequency encoding to handle categorical values
    aza_fre_mapPFX = df.aza.value_counts().to_dict()
    org_code_mapPFX = df.org_code.value_counts().to_dict()
    channel_mapPFX = df.channel.value_counts().to_dict()
    reg_mapPFX = df.reg_mode.value_counts().to_dict()
    sourceoffund_mapPFX = df.source_of_fund.value_counts().to_dict()
    source_mapPFX = df.source.value_counts().to_dict()

    df.aza = df.aza.map(aza_fre_mapPFX)
    df.org_code = df.org_code.map(org_code_mapPFX)
    df.channel = df.channel.map(channel_mapPFX)
    df.reg_mode = df.reg_mode.map(reg_mapPFX)
    df.source_of_fund = df.source_of_fund.map(sourceoffund_mapPFX)
    df.source = df.source.map(source_mapPFX)


    # Label Encoding to handle categorical value
    CUST_TYPE_MAPPFF = {'PFX': 1}
    NOTIFICATION_MAPPFX = {'NOTIFY_SMS_EMAIL': 1}

    df['CUST_TYP_Ordinal'] = df.cust_type.map(CUST_TYPE_MAPPFF)
    df['NOTIFICATION_TYPE_Ordinal'] = df.notification_type.map(NOTIFICATION_MAPPFX)

    # list of numerical
    numerical_featuresdf = [feature for feature in df.columns if df[feature].dtypes != 'O']
    df_Num = df[numerical_featuresdf]

    df_NUM_R = df_Num[['aza', 'IsClientConverted', 'org_code', 'SellingCurrency', 'channel',
        'BuyingCurrency', 'SellingAmount', 'BuyingAmount',
       'BaseCurrency', 'BaseSellingAmount', 'BaseBuyingAmount',
       'trade_contact_id', 'reg_mode',
       'source_of_fund', 'SourceApplication', 'Purpose', 'source', 'Age',
       'Gender_Male', 'trade_acc_id', 'FrequencyOfVisit', 'TimeSpent_seconds',
       'Cluster','Margin']]


    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    create_directories([prepared_data_dir_path])
    prepare_data_path = os.path.join(prepared_data_dir_path, artifacts["PREPARE_DATA"])
    df_NUM_R.to_csv(prepare_data_path, index=False)
    
    #Reading the prepared data from the preparedata folder
    #prepare_datadir = config["artifacts/prepared"]
    prepare_data = os.path.join(prepared_data_dir_path, artifacts["PREPARE_DATA"])
    df_prepare = pd.read_csv(prepare_data, sep=",")


    #Reading the parameters
    split = params["prepare"]["split"]
    seed = params["prepare"]["seed"]
    mlflow.log_params(params["prepare"])
    random.seed(seed)

    #Performing Train-test Split
    train, test = train_test_split(df_prepare, test_size=split,random_state=seed)
    # train_x = train.drop(['Margin'], axis=1, inplace=True)
    # test_y = test.drop(['Margin'], axis=1, inplace=True)
    # train_y = train[["Margin"]]
    # test_y = test[["Margin"]]

    training_test_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["TRAIN_TEST_DATA"])
    create_directories([training_test_data_dir_path])
    
    #Storing the Train-Test Data
    train_data_path = os.path.join(training_test_data_dir_path, artifacts["TRAIN_DATA"])
    train.to_csv(train_data_path, index=False)
    test_data_path = os.path.join(training_test_data_dir_path, artifacts["TEST_DATA"])
    test.to_csv(test_data_path, index=False)



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