# hmm_forrest_nlp

## 1. Create the enviroment.

```
pip install -r requirements.txt
```

## 2. Preprease the data.
Please download the intermediate data used for data analysis in the google drive: 
https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t?usp=sharing

Unzip the tmp_data_open.zip, and use its path to replace the **tmp_dir** variable in the scripts.

Unzip the hmmlearn.zip, and use its path to **path_to_raw_hmm_models** in the scripts.

## 3. Code structure.

1. The jupter notebooks files in the root are the data analysis scripts used to analyze trained HMM models.
2. The original data preprocessing scripts are in the preprocessing folder. The studyforrest data needs to be downloaded to run the script.
3. The training scripts for HMM models are in the training folder. We recommend you to run them on servers with multiple threads.  Please run
'''
python run.py -h 
'''
For the help of specifying different argument

