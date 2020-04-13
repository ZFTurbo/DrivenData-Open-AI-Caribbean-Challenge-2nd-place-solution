# DrivenData: Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery  (2nd place solution)

Repository contains code for solution which won 2nd place in [DrivenData: Open AI Caribbean Challenge: Mapping Disaster Risk from Aerial Imagery](https://www.drivendata.org/competitions/58/disaster-response-roof-type/) competition.

## Software Requirements

Main requirements: Python 3.5+, keras 2.2+, Tensorflow 1.13+, classification_models (latest from git), efficientnet (latest from git)
Other requirements: numpy, pandas, opencv-python, scipy, sklearn, pyvips, pyproj, geopandas, pathlib, shapely
You need to have CUDA 10.0 installed
Solution was tested on Anaconda3-2019.10-Linux-x86_64.sh: https://www.anaconda.com/distribution/

## Hardware requirements

* All batch sizes for Neural nets are tuned to be used on NVIDIA GTX 1080 Ti 11 GB card. To use code with other GPUs with less memory - decrease batch size accordingly.
* At some point during image cache generation code temporary could require around 128 GB of RAM memory (probably swap could handle this).

## How to run:

Code expects all input files in "input/" directory. Fix paths in a00_common_functions.py if needed.
All r*.py files must be run one by one. All intermediate folders will be created automatically.

### Full pipeline:
```
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py
python cnn_v1_densenet121/r16_classification_d121_train_kfold_224.py
python cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py
python cnn_v2_irv2/r16_classification_irv2_train_kfold_299.py
python cnn_v2_irv2/r17_classification_irv2_train_kfold_299_v2.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299_v2.py
python cnn_v3_efficientnet_b4/r17_classification_efficientnet_train_kfold_380.py
python cnn_v3_efficientnet_b4/r26_classification_efficientnet_valid.py
python cnn_v4_densenet169/r17_classification_densenet169_train_kfold_224.py
python cnn_v4_densenet169/r26_classification_densenet169_valid.py
python cnn_v5_resnet34/r17_classification_train_kfold_224.py
python cnn_v5_resnet34/r26_classification_valid.py
python cnn_v6_seresnext50/r17_classification_train_kfold_224.py
python cnn_v6_seresnext50/r26_classification_valid.py
python cnn_v7_resnet50/r17_classification_train_kfold_224.py
python cnn_v7_resnet50/r26_classification_valid.py
python gbm_classifiers/r15_run_catboost.py
python gbm_classifiers/r16_run_xgboost.py
python gbm_classifiers/r17_run_lightgbm.py
python r20_ensemble_avg.py
```

### Only inference part (starting from neural net models):
```
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py
python cnn_v1_densenet121/r26_classification_d121_valid_kfold_224.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299.py
python cnn_v2_irv2/r26_classification_irv2_valid_kfold_299_v2.py
python cnn_v3_efficientnet_b4/r26_classification_efficientnet_valid.py
python cnn_v4_densenet169/r26_classification_densenet169_valid.py
python cnn_v5_resnet34/r26_classification_valid.py
python cnn_v6_seresnext50/r26_classification_valid.py
python cnn_v7_resnet50/r26_classification_valid.py
python gbm_classifiers/r15_run_catboost.py
python gbm_classifiers/r16_run_xgboost.py
python gbm_classifiers/r17_run_lightgbm.py
python r20_ensemble_avg.py
```

**There is file run_inference.sh - which do all the stuff including pip installation of required modules etc.** It was independently tested on fresh system installation from scratch.

Change this variable to location of your python (Anaconda)
`export PATH="/var/anaconda3-temp/bin/"`
Change this vairable to location of your code
`export PYTHONPATH="$PYTHONPATH:/var/test_caribean/"`

### Files needed for inference

* [Neural net weights (~4.4 GB)](https://github.com/ZFTurbo/DrivenData-Open-AI-Caribbean-Challenge-2nd-place-solution/releases)
* [KFold splits](https://github.com/ZFTurbo/DrivenData-Open-AI-Caribbean-Challenge-2nd-place-solution/releases)

### Notes about a code

1) Change "ONLY_INFERENCE" constant in a00_common_functions.py to True for inference without training. You need to use the same
KFold splits, which will be read from cache. You need to use my KFold splits in modified_data folder - it needed to generate
the same feature files as mine for 2nd level models. Because for any other split as it was for training validation files which used in 2nd level model
as input data will be invalid.
2) Python files within different cnn_* folders can be run in parallel. E.g. you can train different neural networks independently.
3) It's only 3 networks enough for good accuracy: cnn_v1_densenet121, cnn_v2_irv2, cnn_v3_efficientnet_b4. All others improve overall result at local validation but lead to the same score on LB.

## File sizes and content
After running:
```
python data_preprocessing/r01_extract_image_data.py
python data_preprocessing/r02_find_neighbours.py
```

You must have the following:
* Folder "modified_data/train_img/" - must contain 45106 images
* Folder "modified_data/test_img/" - must contain 14650 images
* File "modified_data/test.csv" - 3353297 bytes
* File "modified_data/train.csv" - 11564407 bytes

## References

Detailed description of method available at https://arxiv.org/