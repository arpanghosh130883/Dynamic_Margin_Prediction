import black
from pathlib import Path
from black import FileMode, format_file_contents, format_file_in_place, diff, WriteBack


stage_01 = "src/stage_01_dataprepare.py"
stage_02 = "src/stage_02_modeltraining.py"
stage_03 = "src/stage_03_modelevaluation.py"
pytest_testing = "src/pytest_script.py"
black_formatting = "src/black_formatting.py"
vulture_coding = "src/vulture_stdcoding.py"

stages = [
    stage_01,
    stage_02,
    stage_03,
    pytest_testing,
    black_formatting,
    vulture_coding,
]

files_reformatted = 0

for stage in stages:
    path = Path(stage)
    format_file_result = format_file_in_place(
        path, fast=False, mode=FileMode(), write_back=WriteBack.YES
    )
    if format_file_result == True:
        files_reformatted += 1
        print(stage, "reformatted")

print("Files reformatted: ", files_reformatted)
