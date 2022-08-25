import vulture
from pathlib import Path
import os

v = vulture.Vulture()

# path = os.path.normcase('src/sql2.py')

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

for stage in stages:
    path = os.path.normcase(stage)
    v.scavenge([path])

    for item in v.get_unused_code():
        print(item.filename, item.name, item.typ, item.first_lineno)
