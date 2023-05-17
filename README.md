# README

This github repo is for the paper **Bringing language to dynamic brain states: the default network dominates neural responses to evolving movie stories**
## 1. Create the enviroment.

```
pip install -r requirements.txt
```

## 2. Prepare the data.
Please download the intermediate data used for data analysis in the google drive: 
https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t?usp=sharing

Unzip the tmp_data_open.zip, and use its path to replace the variable **tmp_dir** in the scripts.

Unzip the hmmlearn.zip, and use its path to replace the variable **path_to_raw_hmm_models** in the scripts.

## 3. Code structure.

1. The jupter notebooks files in the root are the data analysis scripts used to analyze trained HMM models.
2. The original data preprocessing scripts are in the preprocessing folder. The studyforrest data needs to be downloaded to run the script.
3. The training scripts for HMM models are in the training folder. We recommend you to run them on servers with multiple threads.  For the help of specifying different argument, please run
```
python training/run.py -h 
```

## 4. Demo

Once the intermediate results are downloaded and unziped in the according path. You can run the "single_subject_demonstration.ipynb" for replication of Figs 1-3, and run the "group_analysis.ipynb" for replication of Figs 4-7, and "supplementary_tests.ipynb" for supplementary figues. 
