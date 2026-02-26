import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from structure import *
from config import *
from feature import *

import numpy as np
import time, re, os
from random import shuffle
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

# map feature to get feature function
func_map = {
    "feature_topo_compo": get_feature_topo_compo, 
    "feature_whole_compo": get_feature_whole_compo,
    "feature_add_s_nobin": get_feature_with_s_nobin, 
    "feature_composition": get_feature_composition
}


def learning_cv(data_dir):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_data_X_y(data_dir, fname)
    print("get data")
    
    # normalization
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)

    for n_estimators in n_estimators_arr:
        start = timer()
        clf = GradientBoostingRegressor(loss='squared_error', learning_rate=0.001, n_estimators=n_estimators, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
        # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
        predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
        end = timer()
        
        r2 = r2_score(y, predicted)
        mse = mean_squared_error(y, predicted)
        rmse = np.sqrt(abs(mse))
        mae = mean_absolute_error(y, predicted)

        with open(data_dir + '/predict_' + fname + f'_cv_{n_estimators}.npz', 'wb') as out_file:
            np.savez(out_file, predicted=predicted, y=y, time=end-start, r2=r2, rmse=rmse, mae=mae)

        print(fname + f", cv=10, n_estimator={n_estimators}, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))
        print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))


def learning_cv_repeated(data_dir, fname, times):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_data_X_y(data_dir, fname)
    print("get data")
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    for i in tqdm(range(times), desc="Learning with repeated cv", total=times):
        clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = i)

        # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
        predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
        with open(data_dir + '/predict_' + fname + str(i) + '_rep_cv.npz', 'wb') as out_file:
            np.savez(out_file, predicted=predicted, y=y)
        r2 = r2_score(y, predicted)
        mse = mean_squared_error(y, predicted)
        rmse = np.sqrt(abs(mse))
        mae = mean_absolute_error(y, predicted)

        print(fname + ", cv=10, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))
        print("times: " + str(i))
        print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

def _load_single_feature(args):
    """Helper function to load a single feature file."""
    idx, id, delta_e, data_dir, fname = args
    feature_path = data_dir + '/' + fname + '/' + id + '_feature.npy'
    with open(feature_path, 'rb') as fe:
        feature = np.load(fe)
    return idx, feature, delta_e

def get_data_X_y(data_dir, fname):
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()[1:]  # Skip first datapoint, since it seems to be doing that elsewhere
    
    n_samples = len(lines)
    
    first_id = lines[0].split()[0]
    first_feature_path = data_dir + '/' + fname + '/' + first_id + '_feature.npy'
    with open(first_feature_path, 'rb') as fe:
        first_feature = np.load(fe)
    feature_dim = first_feature.shape[0]
    
    # Prepare output arrays
    X = np.zeros((n_samples, feature_dim), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    
    # Prepare arguments for parallel loading
    load_args = []
    for idx, line in tqdm(enumerate(lines), desc="Preparing load arguments", total=n_samples):
        id, delta_e = line.split()[0], line.split()[1]
        load_args.append((idx, id, delta_e, data_dir, fname))
    
    # Load features in parallel
    with Pool(cpus) as pool:
        results = list(tqdm(
            pool.imap(_load_single_feature, load_args),
            desc="Loading features",
            total=n_samples
        ))
    
    # Place results in correct positions
    for idx, feature, delta_e in tqdm(results, desc="Placing results", total=n_samples):
        X[idx] = feature
        y[idx] = float(delta_e)
    
    return X, y


def get_id_list(data_dir):
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    id_list = []

    for line in tqdm(lines[1:], desc="Getting id list", total=len(lines)-1):
        id = line.split()[0]
        id_list.append(id)
    return id_list


from timeit import default_timer as timer

def batch_handle(id_list):
    for id in id_list:
        get_prim_structure_info(data_dir, id)
        cav, cev = enlarge_cell(data_dir, id)
        all_pair_outs = get_betti_num(data_dir, id, cav, cev)
        whole_bettis_out = get_betti_whole_lattice(data_dir, id, cav, cev)
        func_map[fname](data_dir, id, cav, cev, all_pair_outs, whole_bettis_out)


def split_list(all_id_list):
    id_list_splited = []
    step = math.ceil(len(all_id_list)/10.0)
    for i in range(0, len(all_id_list), step):
        start = i
        end = min(i + step, len(all_id_list))
        id_list_splited.append(all_id_list[start: end])
    # print(id_list_splited[0][0], id_list_splited[1][0])
    return id_list_splited
