import numpy as np
from scipy.stats import pearsonr
import nilearn
from nilearn import plotting
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import nibabel
import re
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sns
from datetime import date
import time
import os
import itertools
import argparse
from scipy import ndimage
from scipy.stats import zscore
from sklearn.decomposition import PCA

def fmri_real_time_map(x):
    if x > 21 * 60 + 32.12:
        x = x + 24*60 + 13.24 - 21*60+32.12
    if x > 38 * 60 +31.23:
        x = x + 38 * 60 +58.20 - (38 * 60 +31.23)
    if x > 57 * 60 + 19.22:
        x = x + 59 * 60 + 31.17 - (57 * 60 + 19.22)
    if x > 1 * 3600 + 18 * 60 + 14.00:
        x = x + 3600 + 20 * 60 + 24.16 - (3600 + 18 * 60 + 14.00)
    if x > 3600 + 34 * 60 + 18.06:
        x = x + 3600 + 37 * 60 + 14.19 - (3600 + 34 * 60 + 18.06)
    if x > 3600 + 41 * 60 + 30.19:
        x = x + 3600 + 42 * 60 + 49.19 - (3600 + 41 * 60 + 30.19)
    return x

def denoise_brain(brain_maps):
    brain_maps_data = nilearn.image.get_data(brain_maps)
    brain_mask = brain_maps_data[..., 0].astype(bool)
    brain_mask = ndimage.binary_erosion(brain_mask)
    brain_mask = ndimage.binary_dilation(brain_mask)
    brain_maps_data[~brain_mask] = 0
    brain_map_after = nibabel.nifti1.Nifti1Image(
        brain_maps_data, affine=brain_maps.affine)
    return brain_map_after


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    if 'args.npy' in listOfFile:
        allFiles.append(dirName+'/args.npy')
        return allFiles
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif fullPath[-8:] == "args.npy":
            allFiles.append(fullPath)

    return allFiles


def filter_args(prefix, al):
    #     return [i.replace("data", "data_57") for i in al if prefix not in i]
    return [
        i.replace("_list", "") for i in al
        if len(re.findall(prefix, i)) is not 0
    ]


def parse_path(arg_path):
    arg_path = arg_path.replace("DMN_smith", "16_smith")
    strinfo = re.compile("HMMs_\d\d_\d\d_\d\d")
    return strinfo.sub(
        "",
        arg_path.split("/")[-3:-1][0]) + arg_path.split("/")[-3:-1][1].replace(
            "subject_01", "")


def get_xticks(k):
    n_pcs_dict = {'nac': 2, 'hippo': 6, 'amyg': 6, 'all': 14}
    nac_type = re.findall("(nac|hippo|amyg|all)", k)[0]
    region = re.findall(
        "(Vis|SomMot|DorsAttn|SalVentAttn|Limbic|Cont|Default|16)", k)[0]
    rmv = int(n_pcs_dict[nac_type] / 2)
    schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(100)
    labels_strings = []
    for label in schaefer.labels:
        labels_strings.append(label.decode('utf-8'))
    xticks = [
        i.replace('7Networks_', '') for i in labels_strings if region in i
    ] + [i + nac_type for i in ['left_'] * rmv + ['right_'] * rmv]
    return xticks



def get_args(l, pls=False, cca=0, inp=None, return_idx=0):
    yeo7 = ['Vis', 'SomMot', 'DorsAttn',
        'SalVentAttn', 'Limbic', 'Cont', 'Default']
    annotations = np.load('/Users/enningyang/Documents/forrest_project/tmp_data/annotations.pickle', allow_pickle=True)
    annotations_keys = list(annotations.keys())
    nac_type = re.findall("(nac|hippo|amyg|all)", l)[0]
    region = re.findall("(Vis|SomMot|DorsAttn|SalVentAttn|Limbic|Cont|Default|16)", l)[0]
    n_components = re.findall("n(\d+)_", l)[0]
    text_type = re.findall("states_(\w+)\/", l)[0]
    text_method = text_type.split('_')[1]
    text_type = text_type.split('_')[0]
    add_pcs = re.findall("(\d)pcs", l)[0]
    subject = re.findall("subject_(\d+)_", l)[0]
    in_type = re.findall(f"{region}\_(.*)\_{nac_type}", l)[-1]

    add_text = 'add_text' in l
    
    if nac_type == "hippo":
        if not add_pcs:
            add_pcs = 3
        nac_timeseries_dict = pickle.load(
            open("/Users/enningyang/Documents/forrest_project/tmp_data/hippo_timeseries_{}pcs_dict.pickle".format(add_pcs),
                 "rb"))
    elif nac_type == "amyg":
        if not add_pcs:
            add_pcs = 3
        nac_timeseries_dict = pickle.load(
            open("/Users/enningyang/Documents/forrest_project/tmp_data/amyg_timeseries_{}pcs_dict.pickle".format(add_pcs),
                 "rb"))
    elif nac_type == "all":
        nac_timeseries_dict1 = pickle.load(
            open("/Users/enningyang/Documents/forrest_project/tmp_data/hippo_timeseries_{}pcs_dict.pickle".format(3),
                 "rb"))
        nac_timeseries_dict2 = pickle.load(
            open("/Users/enningyang/Documents/forrest_project/tmp_data/amyg_timeseries_{}pcs_dict.pickle".format(3), "rb"))
        nac_timeseries_dict = {}
        for k, v in nac_timeseries_dict1.items():
            nac_timeseries_dict[k] = np.hstack(
                [v, nac_timeseries_dict2[k]])
        add_pcs = 6

    if in_type == "yeo":
        data_dict = pickle.load(
            open("/Users/enningyang/Documents/forrest_project/tmp_data/schaefer_timeseries_dict_z_scored.pickle", "rb"))
        schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(100)
        labels_strings = []
        for label in schaefer.labels:
            labels_strings.append(label.decode('utf-8'))
        if region == "whole_brain":
            idx = range(100)
        else:
            idx = [i for i, s in enumerate(labels_strings) if region in s]
    else:
        data_dict = pickle.load(
            open(f"{tmp_dir}/schaefer_timeseries_dict_{in_type}.pickle", "rb"))
        idx = range(yeo7.index(region)*5, yeo7.index(region)*5+5)
        

    text = pickle.load(
        open(
            "/Users/enningyang/Documents/forrest_project/tmp_data/text_embeddings_n{}_{}_{}.pickle".format(
                n_components, text_type, text_method), "rb"))
    text_keys = list(text.keys())
    text_components = text[text_keys[0]].shape[1]

    if idx:
        brain = data_dict[subject][:, idx]
    else:
        brain = data_dict[subject]
    accumbens_timeseries = nac_timeseries_dict[subject]
    X_stack = np.hstack((brain, accumbens_timeseries))
    if add_text:
        X_stack = np.hstack((X_stack, text[240]))

    HMMs_subject_path = l[:-8]

    try:
        sorted_text_score = pickle.load(
            open(os.path.join(HMMs_subject_path, "sorted_text_score.pickle"),
                 "rb"))
        if return_idx:
            state_idx = pickle.load(
                open(os.path.join(HMMs_subject_path, "state_idx.pickle"),
                    "rb"))
        top_3_text_score = dict(itertools.islice(sorted_text_score.items(), 3))
        model_count = list(top_3_text_score.keys())[0]
        # print(l, model_count)
    except Exception as e:
        print(e)
        return

    file_path = os.path.join(HMMs_subject_path,
                             'HMM_r{}'.format(model_count) + '.pickle')
    HMM = pickle.load(open(file_path, 'rb'))
    if return_idx:
        idx = state_idx[model_count]
    state_probabilities_timeseries = HMM.predict_proba(X_stack).T
    transition_matrix = HMM.transmat_

    if not pls and not cca:
        cor_annotation = np.zeros(
            shape=(len(annotations), state_probabilities_timeseries.shape[0]))
        for annotation in range(len(annotations)):
            for state in range(state_probabilities_timeseries.shape[0]):
                cor_annotation[annotation, state] = pearsonr(
                    annotations[annotations_keys[annotation]],
                    state_probabilities_timeseries[state])[0]
        cor_text = np.zeros(shape=(len(text), text_components,
                                   state_probabilities_timeseries.shape[0]))
        for l in range(len(text)):
            for state in range(state_probabilities_timeseries.shape[0]):
                for i in range(text_components):
                    cor_text[l, i, state] = pearsonr(
                        text[text_keys[l]][:, i],
                        state_probabilities_timeseries[state])[0]
    elif pls:
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=pls)
        X = state_probabilities_timeseries.T
        Y1 = np.array(list(annotations.values())).T
        m_list = []
        model.fit(X, Y1)
        m_list.append(model)
        cor_annotation = model.coef_.T
        cor_text = np.zeros(shape=(len(text), text_components,
                                   state_probabilities_timeseries.shape[0]))
        for i, k in enumerate(text.keys()):
            model.fit(X, text[k])
            m_list.append(model)
            cor_text[i, ...] = model.coef_.T
        if return_idx:
            return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack,idx,m_list
        else:   
            return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack, m_list
    elif cca:
        from sklearn.cross_decomposition import CCA
        model = CCA(n_components=pls)
        X = state_probabilities_timeseries.T
        Y1 = np.array(list(annotations.values())).T
        m_list = []
        model.fit(X, Y1)
        m_list.append(model)
        cor_annotation = model.coef_.T
        cor_text = np.zeros(shape=(len(text), text_components,
                                   state_probabilities_timeseries.shape[0]))
        for i, k in enumerate(text.keys()):
            model.fit(X, text[k])
            m_list.append(model)
            cor_text[i, ...] = model.coef_.T
        if return_idx:
            return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack,idx,m_list
        else:   
            return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack, m_list
    
        

    if return_idx:
        return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack,idx
    else:   
        return state_probabilities_timeseries, cor_annotation, cor_text, HMM, annotations_keys, text_keys, X_stack,



def pre_box(d):
    c1 = []
    c2 = []
    for k, v in d.items():
        c1.append([k + 1] * len(v))
        c2.append(v)
    return {"keys": np.hstack(c1), "values": np.hstack(c2)}


def cal_len(a, target):
    rst = []
    c = 0
    for i in a:
        if i != target:
            if c > 0:
                rst.append(c)
            c = 0
            continue
        c += 1
    if c > 0:
        rst.append(c)
    return rst


def cal_len_con(df, target, col, cond):
    df_day = df.loc[df[cond] == 1, :]
    list_of_df = np.split(df_day,
                          np.flatnonzero(np.diff(df_day.index) != 1) + 1)
    rst_col = {}
    for t in target:
        rst = []
        for sub_df in list_of_df:
            rst.append(cal_len(sub_df[col].to_list(), t))
        rst_col[t] = np.hstack(rst)
    return rst_col


def fit_schaefer(data, region):

    schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(100)
    labels_strings = []
    for label in schaefer.labels:
        labels_strings.append(label.decode('utf-8'))
    idx = [i for i, s in enumerate(labels_strings) if region in s]
    states_map = np.zeros([1, 100])
    states_map[0, idx] = zscore(data)
    maps = schaefer.maps
    labels = schaefer.labels
    masker = NiftiLabelsMasker(labels_img=maps,
                               standardize=True,
                               memory='nilearn_cache').fit()
    masker_aligned = NiftiLabelsMasker(
        labels_img='/home/enning/projects/rrg-danilobz/enning/datasets/forrest/aligned_data_fits/Schaefer_100/schaefer_100_maps_rounded_subject_01.nii',
        standardize=True,
        memory='nilearn_cache').fit()
    mean_map = masker_aligned.inverse_transform(states_map)
    return mean_map


def prepare_masks():
    right_hippoAma = nilearn.image.load_img(
        '/home/enning/projects/rrg-danilobz/enning/datasets/forrest/masks/hippoAmygLabels/sub01_rh.nii.gz'
    )
    left_hippoAma = nilearn.image.load_img(
        '/home/enning/projects/rrg-danilobz/enning/datasets/forrest/masks/hippoAmygLabels/sub01_lh.nii.gz'
    )

    right_hippoAma_data = nilearn.image.get_data(right_hippoAma)
    right_hippoAma_data = np.around(right_hippoAma_data).astype(int)
    right_hippo_data = right_hippoAma_data.copy()
    right_amyg_data = right_hippoAma_data.copy()
    right_hippo_data[right_hippo_data > 7000] = 0
    right_amyg_data[right_amyg_data < 7000] = 0

    left_hippoAma_data = nilearn.image.get_data(left_hippoAma)
    left_hippoAma_data = np.around(left_hippoAma_data).astype(int)
    left_hippo_data = left_hippoAma_data.copy()
    left_amyg_data = left_hippoAma_data.copy()
    left_hippo_data[left_hippo_data > 7000] = 0
    left_amyg_data[left_amyg_data < 7000] = 0

    left_hippo_maps = nibabel.nifti1.Nifti1Image(left_hippo_data,
                                                 affine=left_hippoAma.affine)
    right_hippo_maps = nibabel.nifti1.Nifti1Image(right_hippo_data,
                                                  affine=right_hippoAma.affine)
    left_amyg_maps = nibabel.nifti1.Nifti1Image(left_amyg_data,
                                                affine=left_hippoAma.affine)
    right_amyg_maps = nibabel.nifti1.Nifti1Image(right_amyg_data,
                                                 affine=right_hippoAma.affine)

    lh_masker = NiftiLabelsMasker(labels_img=left_hippo_maps,
                                  standardize=True,
                                  memory='nilearn_cache').fit()
    rh_masker = NiftiLabelsMasker(labels_img=right_hippo_maps,
                                  standardize=True,
                                  memory='nilearn_cache').fit()
    la_masker = NiftiLabelsMasker(labels_img=left_amyg_maps,
                                  standardize=True,
                                  memory='nilearn_cache').fit()
    ra_masker = NiftiLabelsMasker(labels_img=right_amyg_maps,
                                  standardize=True,
                                  memory='nilearn_cache').fit()
    ln_masker = np.load(
        '/home/enning/scratch/forrest_group/tmp_data/lh_masker.npy', allow_pickle=1).item().fit()
    rn_masker = np.load(
        '/home/enning/scratch/forrest_group/tmp_data/rh_masker.npy', allow_pickle=1).item().fit()

    masker_dict = {
        "nac": [ln_masker, rn_masker],
        "hippo": [lh_masker, rh_masker],
        "amyg": [la_masker, ra_masker]
    }
    return masker_dict

def sort_state(HMM1, HMM2,n_states=4):
    from scipy.optimize import linear_sum_assignment
    HMM_com = HMM1
    paras_com = []
    for i in range(n_states):
        paras_com.append(np.hstack([HMM_com.means_[i,:24],np.ravel(HMM_com.covars_[i,:24,:24])]))

    paras = []
    HMM = HMM2
    for i in range(n_states):
        paras.append(np.hstack([HMM.means_[i,:24],np.ravel(HMM.covars_[i,:24,:24])]))
    
    cost_matrix = np.zeros([n_states,n_states])
    for i in range(n_states):
        for j in range(n_states):
            cost_matrix[i, j] = pearsonr(paras_com[i],paras[j])[0]

    ri,ci = linear_sum_assignment(-1*cost_matrix)
    return ci

def fetch_input(l,tmp_dir):
    nac_type = re.findall("(nac|hippo|amyg|all)", l)[0]
    region = re.findall("(Vis|SomMot|DorsAttn|SalVentAttn|Limbic|Cont|Default|16)", l)[0]
    n_components = re.findall("n(\d+)_", l)[0]
    text_type = re.findall("states_(\w+)\/", l)[0]
    text_method = text_type.split('_')[1]
    text_type = text_type.split('_')[0]
    add_pcs = re.findall("(\d)pcs", l)[0]
    subject = re.findall("subject_(\d+)_", l)[0]
    in_type = re.findall(f"{region}\_(.*)\_{nac_type}", l)[-1]
    data_dict = pickle.load(
        open(f"{tmp_dir}/schaefer_timeseries_dict_z_scored.pickle", "rb"))
    schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(100)
    labels_strings = []
    for label in schaefer.labels:
        labels_strings.append(label.decode('utf-8'))
    idx = [i for i, s in enumerate(labels_strings) if region in s]
    if nac_type == "hippo":
        nac_timeseries_dict = pickle.load(
            open(f"{tmp_dir}/hippo_timeseries_3pcs_dict.pickle", "rb"))
    else:
        nac_timeseries_dict = pickle.load(
            open(f"{tmp_dir}/amyg_timeseries_3pcs_dict.pickle", "rb"))
    if idx:
        brain = data_dict[subject][:, idx]
    else:
        brain = data_dict[subject]
    accumbens_timeseries = nac_timeseries_dict[subject]
    X_stack = np.hstack((brain, accumbens_timeseries))
    return X_stack

def bic_general(likelihood_fn, k, X, AIC=False):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    bic = np.log(len(X))*k - 2*likelihood_fn(X)
    if AIC:
        bic = 2*k-2*likelihood_fn(X)
    return bic

def bic_hmmlearn(hmm_curr, X, AIC=False):
    lowest_bic = np.infty
    bic = []
    n_components = hmm_curr.n_components
    # Calculate number of free parameters
    # free_parameters = for_means + for_covars + for_transmat + for_startprob
    # for_means & for_covars = n_features*n_components
    n_features = hmm_curr.n_features-6
    free_parameters = n_components*n_features+ n_components * n_features * (n_features - 1) / 2 + n_components*(n_components-1) + (n_components-1)

    bic_curr = bic_general(hmm_curr.score, free_parameters, X, AIC)

    return bic_curr