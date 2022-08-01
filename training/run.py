from hmmlearn import hmm
import numpy as np
from scipy.stats import pearsonr
import nilearn
from nilearn import datasets
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import time
import os
import sys
import itertools
import argparse
from multiprocessing import Pool
from sklearn.decomposition import PCA
from scipy import ndimage
import warnings

# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)
print('Start')
yeo7 = ['Vis', 'SomMot', 'DorsAttn',
        'SalVentAttn', 'Limbic', 'Cont', 'Default']

parser = argparse.ArgumentParser()
parser.add_argument("--subject", "-s", type=str, default="01",help="select the subject")
parser.add_argument("--n_components", "-t", type=int, default=200, help="number of semantic contexts")
parser.add_argument("--n_models", "-n", type=int, default=100, help="number of HMM models to train")
parser.add_argument("--n_states", type=int, default=4, help="number of brian states")
parser.add_argument("--text_type", type=str, default="sub",help="choose from sub(subtitles) or des(descriptions)")
parser.add_argument("--text_method", type=str, default="nmf", help="choose from nmf or pca (vanila LSA)")
parser.add_argument("--region", default="DMN",help="hoose from 'Vis', 'SomMot', 'DorsAttn','SalVentAttn', 'Limbic', 'Cont','Default'")
parser.add_argument("--limbictype", type=str, default="hippo", help="The appended limbic regions: choose from hippo or amyg")
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--path", type=str, default=None, help='the path which stored all the input files')

n_pcs_dict = {'hippo': 6, 'amyg': 6}
skip = 0
subject_compare = '01' # The traget subject for hungarian algorithm
args = parser.parse_args()
tmp_dir = args.path
suffix = args.suffix
limbic_type = args.limbictype
if limbic_type == "hippo":
    add_pcs = 3
    limbic_timeseries_dict = pickle.load(
        open(f"{tmp_dir}/hippo_timeseries_{add_pcs}pcs_dict.pickle", "rb"))
elif limbic_type == "amyg":
    add_pcs = 3
    limbic_timeseries_dict = pickle.load(
        open(f"{tmp_dir}/amyg_timeseries_{add_pcs}pcs_dict.pickle", "rb"))
else:
    raise error("no such limbic type")


region = args.region
in_type = args.datatype
data_dict = pickle.load(
    open(f"{tmp_dir}/schaefer_timeseries_dict_z_scored.pickle", "rb"))
schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(100)
labels_strings = []
for label in schaefer.labels:
    labels_strings.append(label.decode('utf-8'))
if region == "whole_brain":
    idx = range(100)
else:
    idx = [i for i, s in enumerate(labels_strings) if region in s]


subjects = [args.subject]
n_components = args.n_components
n_models = args.n_models
n_states = args.n_states

#load annotaions and semantic contexts
annotations = pickle.load(open(f"{tmp_dir}/annotations.pickle", "rb"))
annotations_keys = list(annotations.keys())
text_type = args.text_type
text_method = args.text_method
if os.path.exists("{}/text_embeddings_s{}_n{}_{}_{}.pickle".format(tmp_dir, args.subject, n_components, text_type, text_method)):
    text = pickle.load(open("{}/text_embeddings_s{}_n{}_{}_{}.pickle".format(
        tmp_dir, args.subject, n_components, text_type, text_method), "rb"))
else:
    text = pickle.load(open("{}/text_embeddings_n{}_{}_{}.pickle".format(
        tmp_dir, n_components, text_type, text_method), "rb"))

text_keys = list(text.keys())
text_components = text[text_keys[0]].shape[1]

today = date.today()
if not suffix:
    suffix = today.strftime("%d_%m_%y")

try:
    HMMs_path = os.path.join("/home/enning/scratch/forrest/hmmlearn", 'HMMs_' + suffix +
                             '_{}_{}_{}_{}pcs'.format(region, in_type, limbic_type, add_pcs))
    os.mkdir(HMMs_path)

except OSError as error:
    pass
#         print("oserror", error)

state_annotations_scores_dict = {}
global_start = time.time()
subject = subjects[0]

try:
    HMMs_subject_path = os.path.join(HMMs_path, 'subject_{}_n{}_{}_states_{}_{}'.format(
        subject, text_components, n_states, text_type, text_method))
    os.mkdir(HMMs_subject_path)

except OSError as error:
    pass

try:
    tmp = pickle.load(
        open(os.path.join(HMMs_subject_path, 'sorted_text_score.pickle'), 'rb'))
    if len(list(tmp.keys())) != 0:
        print("has been runned.\n")
except OSError as error:
    pass


# prepare the training data                  
if idx:
    brain = data_dict[subject][:, idx]
else:
    brain = data_dict[subject]
limbic_timeseries = limbic_timeseries_dict[subject]
X_stack = np.hstack((brain, limbic_timeseries))

state_annotations_scores = {}
state_text_scores = {}
random_state = 0 
model_counter = n_models

# train different models 
while model_counter > 0:
    # initialize the model
    HMM = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                          random_state=random_state, n_iter=500)
    HMM.fit(X_stack)
    transition_matrix = HMM.transmat_

    # making sure that the model did not fit an identity matrix
    # change to the desired threshold
    if np.all(np.diag(transition_matrix) < 0.97):
        print(np.diag(transition_matrix))
        with open(os.path.join(HMMs_subject_path, 'HMM_r{}'.format(random_state) + '.pickle'), 'wb') as handle:
            pickle.dump(HMM, handle)

        state_probabilities_timeseries = HMM.predict_proba(X_stack).T

        """
        model score based on the correlation between the state probability and the annotations timeseries

        change indexation depending on the time-lagged experiment:
        annotations[annotations_keys[annotation]][1:] or annotations[annotations_keys[annotation]][:-1]
        """
        """
        lack annotation files here
        """
        cor_annotation = np.zeros(
            shape=(len(annotations), state_probabilities_timeseries.shape[0]))

        for annotation in range(len(annotations)):
            for state in range(state_probabilities_timeseries.shape[0]):
                cor_annotation[annotation, state] = pearsonr(
                    annotations[annotations_keys[annotation]], state_probabilities_timeseries[state])[0]
        state_annotations_scores[random_state] = np.sum(
            np.ravel(np.abs(cor_annotation)))

        cor_text = np.zeros(shape=(len(text), text_components,
                                   state_probabilities_timeseries.shape[0]))
        for ll in range(len(text)):
            for state in range(state_probabilities_timeseries.shape[0]):
                for i in range(text_components):
                    cor_text[ll, i, state] = pearsonr(
                        text[text_keys[ll]][:, i], state_probabilities_timeseries[state])[0]
        state_text_scores[random_state] = np.sum(np.ravel(np.abs(cor_text)))
        model_counter -= 1

    # catching the somewhat frequent LinAlgError and UFuncTypeError
    # We should make sure that no other error is caught.

    random_state += 1
    if random_state > 200:
        if model_counter == n_models:
            raise Exception("HMM not converges with " + str(args))
        else:
            print(args, model_counter)
            skip = 1
            break

state_annotations_scores_dict[subject] = state_annotations_scores
subject_end = time.time()

# save the best model and model scores                   
if not skip:
    sorted_text_score = {key: value for key, value in sorted(
        state_text_scores.items(), key=lambda item: item[1], reverse=True)}
    #         print(sorted_text_score)

    with open(os.path.join(HMMs_subject_path, 'sorted_text_score.pickle'), 'wb') as handle:
        pickle.dump(sorted_text_score, handle)
# save the data
with open(os.path.join(HMMs_subject_path, 'args.npy'), "wb") as handle:
    pickle.dump(args, handle)
if os.path.exists(f"{tmp_dir}/hmmlearn_args_list.npy"):
    args_list = list(np.load(f"{tmp_dir}/hmmlearn_args_list.npy"))
    args_list.append(os.path.join(HMMs_subject_path, 'args_list.npy'))
    np.save(f"{tmp_dir}/hmmlearn_args_list.npy", args_list)
else:
    np.save(f"{tmp_dir}/hmmlearn_args_list",
            [os.path.join(HMMs_subject_path, 'args.npy')])
    print(l)

print("Training finished")
global_end = time.time()
print('total_time', global_end - global_start)
