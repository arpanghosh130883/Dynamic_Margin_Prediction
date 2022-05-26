import mlflow


def main():
    #mlflow.set_tracking_uri("http://127.0.0.1:1234")


    with mlflow.start_run() as run:
        mlflow.run(".", "stage_01", use_conda=False)
        mlflow.run(".", "stage_02", use_conda=False)
        mlflow.run(".", "stage_03", use_conda=False)
        

      



if __name__ == "__main__":
    main()