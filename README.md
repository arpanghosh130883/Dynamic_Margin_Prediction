# MLflow-project-Dynamic Margin
MLflow Dynamic Margin Prediction

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05 - Create conda.yaml file -
```bash
conda env export > conda.yaml
```

### STEP 06- commit and push the changes to the remote repository

### Command to use MLFlow tracking server
```bash
mlflow server --backend-store-uri sqlite:///mlfow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 1234
```