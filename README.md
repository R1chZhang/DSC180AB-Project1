# DSC180AB-Project1
This project attempts to replicate some of the long-range
graph benchmarks result. The goal is to approaching the original metric numbers as well as
develop an understandig of model performance comparatively across different models and task types.
# Project Outline

-.gitignore

-run.py

-README.md

-test: testData.py

-references

-requirements.txt

-src: data: LRGBDataset.py, graph_level_model.py, model.py

# Access Data

In case the built-in Pytorch geometric modules does not contain the required data module (which can actually happene).

Place the LRGBDataset.py under **envs\<env_name>\Lib\sitepackages\torch_geometric\datasets** to manualy download datasets. 

Reference of LRGB dataset: https://github.com/vijaydwivedi75/lrgb

# Running the Project

run with **python run.py test** to run the test data
and **python run.py all** for the entire project. You may need to manually add dataset to PyG for running the entire project.
