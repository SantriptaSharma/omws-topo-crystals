import numpy as np
import pandas as pd
import os
import math
import re
from config import *

import pickle as pkl

from tqdm import tqdm

def atoms_frequency():
    sub_dict = split_element(data_dir)

    atom_single = {}
    for sub in tqdm(sub_dict.keys(), desc="Calculating atom frequency", total=len(sub_dict)):
        for i in sub_dict[sub]:
            if i not in atom_single:
                atom_single[i] = 0.0
            atom_single[i] += 1.0
    return atom_single

def split_element(data_dir):
    if os.path.exists(data_dir + '/sub_dict.pkl'):
        with open(data_dir + '/sub_dict.pkl', 'rb') as f:
            sub_dict = pkl.load(f)
        return sub_dict

    pattern = re.compile("[A-Z]{1}[a-z]{0,1}")
    sub_dict = {}

    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    for line in tqdm(lines, desc="Splitting elements", total=len(lines)):
        sub = line.split()[0]
        with open(data_dir + "/structure/" + sub, 'r') as tmp:
            ls = tmp.read().splitlines()[0]
        eles = set(pattern.findall(ls))
        sub_dict[sub] = eles

    with open(data_dir + '/sub_dict.pkl', 'wb') as f:
        pkl.dump(sub_dict, f)

    return sub_dict

atom_single = atoms_frequency()

common_pair = {}
i = 0
for key in sorted(atom_single, key=atom_single.__getitem__,reverse=True):
    common_pair[key] = i
    i += 1
com_len = len(common_pair)

element_properties = pd.read_csv(data_dir + '/element_properties.csv')
element_properties.set_index('Abbr', inplace=True)

def get_typ_dict(center_atom_vec):
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1

    return typ_dict

def compute_feature_composition(typ_dict):
    Feature = []; tmp_array = []
    for ele, n in typ_dict.items():
        ele_list = list(element_properties.loc[ele])
        for i in range(n):
            tmp_array.append(ele_list)

    Feature.append(np.mean(tmp_array, axis=0))
    Feature.append(np.std(tmp_array, axis=0))
    Feature.append(np.sum(tmp_array, axis=0))
    Feature.append(np.max(tmp_array, axis=0))
    Feature.append(np.min(tmp_array, axis=0))

    Feature_1 = np.asarray(Feature, float)
    Feature_1 = np.concatenate(Feature_1, axis=0)
    return Feature_1

def get_feature_composition(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis, whole_bettis, alpha_bettis):
    save_path = data_dir + '/feature_composition/' + id + '_feature.npy'
    if os.path.exists(save_path):
        return

    typ_dict = get_typ_dict(center_atom_vec)
    Feature_1 = compute_feature_composition(typ_dict)

    if not os.path.exists(data_dir + "/feature_composition"):
        os.makedirs(data_dir + "/feature_composition", exist_ok=True)

    with open(save_path, 'wb') as outfile:
        np.save(outfile, Feature_1)


def compute_statistics(arr_raw, arr_weighted):
    if len(arr_raw) == 0:
        return [0.]*5
    else:
        return [np.mean(arr_raw), np.std(arr_raw), np.max(arr_raw), np.min(arr_raw), np.sum(arr_weighted)]

def compute_feature_with_s_nobin(pair_bettis, typ_dict):
    bar_births = [[] for _ in range(3)]
    bar_deaths = [[] for _ in range(3)]

    bar_births_weighted = [[] for _ in range(3)]
    bar_deaths_weighted = [[] for _ in range(3)]

    Feature_3 = np.zeros([com_len, 7*5], float)

    for (ctype, eltype, dgms) in pair_bettis:
        ca_num = typ_dict[ctype]
        cp_idx = common_pair.get(ctype, None)
    
        for dim, dgm in enumerate(dgms):

            births = bar_births[dim]
            deaths = bar_deaths[dim]
            Wbirths = bar_births_weighted[dim]
            Wdeaths = bar_deaths_weighted[dim]

            nbirths = [birth for birth, _ in dgm]
            ndeaths = [death if death != float('inf') else -1 for birth, death in dgm]
            n_wbirths = [birth/ca_num for birth, death in dgm]
            n_wdeaths = [(death if death != float('inf') else -1) for birth, death in dgm]

            births.extend(nbirths)
            deaths.extend(ndeaths)
            Wbirths.extend(n_wbirths)
            Wdeaths.extend(n_wdeaths)

            if cp_idx is None:
                continue

            nbirths = np.array(nbirths)
            n_wbirths = np.array(n_wbirths)
            ndeaths = np.array(ndeaths)
            n_wdeaths = np.array(n_wdeaths)
            nlifetimes = ndeaths - nbirths
            n_wlifetimes = n_wdeaths - n_wbirths

            valid_nmask = ndeaths != -1
            valid_nbirths = nbirths[valid_nmask]
            valid_ndeaths = ndeaths[valid_nmask]
            valid_nlifetimes = nlifetimes[valid_nmask]
            valid_n_wbirths = n_wbirths[valid_nmask]
            valid_n_wdeaths = n_wdeaths[valid_nmask]
            valid_n_wlifetimes = n_wlifetimes[valid_nmask]

            if dim == 0:
                Feature_3[cp_idx, 0:5] = np.array(compute_statistics(valid_ndeaths, valid_n_wdeaths))
            elif dim == 1:
                Feature_3[cp_idx, 5:10] = np.array(compute_statistics(valid_nlifetimes, valid_n_wlifetimes))
                Feature_3[cp_idx, 10:15] = np.array(compute_statistics(valid_nbirths, valid_n_wbirths))
                Feature_3[cp_idx, 15:20] = np.array(compute_statistics(valid_ndeaths, valid_n_wdeaths))
            elif dim == 2:
                Feature_3[cp_idx, 20:25] = np.array(compute_statistics(valid_nlifetimes, valid_n_wlifetimes))
                Feature_3[cp_idx, 25:30] = np.array(compute_statistics(valid_nbirths, valid_n_wbirths))
                Feature_3[cp_idx, 30:35] = np.array(compute_statistics(valid_ndeaths, valid_n_wdeaths))

    Feature_3 = np.concatenate(Feature_3, axis=0)

    bars_births = [np.array(bar_births[i]) for i in range(3)]
    bars_deaths = [np.array(bar_deaths[i]) for i in range(3)]
    bars_births_weighted = [np.array(bar_births_weighted[i]) for i in range(3)]
    bars_deaths_weighted = [np.array(bar_deaths_weighted[i]) for i in range(3)]

    Feature_2 = []

    valid_masks = [bars_deaths[i] != -1 for i in range(3)]
    valid_births = [bars_births[i][valid_masks[i]] for i in range(3)]
    valid_deaths = [bars_deaths[i][valid_masks[i]] for i in range(3)]
    valid_births_weighted = [bars_births_weighted[i][valid_masks[i]] for i in range(3)]
    valid_deaths_weighted = [bars_deaths_weighted[i][valid_masks[i]] for i in range(3)]
    lifetimes = [valid_deaths[i] - valid_births[i] for i in range(3)]
    lifetimes_weighted = [valid_deaths_weighted[i] - valid_births_weighted[i] for i in range(3)]

    Feature_2.extend(compute_statistics(valid_deaths[0], valid_deaths_weighted[0]))

    for d in (1, 2):
        Feature_2.extend(compute_statistics(lifetimes[d], lifetimes_weighted[d]))
        Feature_2.extend(compute_statistics(valid_births[d], valid_births_weighted[d]))
        Feature_2.extend(compute_statistics(valid_deaths[d], valid_deaths_weighted[d]))

    Feature_2 = np.asarray(Feature_2, float)

    Feature = np.concatenate((Feature_2, Feature_3), axis=0)
    return Feature

def get_feature_with_s_nobin(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis, whole_bettis, alpha_bettis):
    save_path = data_dir + '/feature_add_s_nobin/' + id + '_feature.npy'
    if os.path.exists(save_path):
        return

    typ_dict = get_typ_dict(center_atom_vec)
    Feature = compute_feature_with_s_nobin(pair_bettis, typ_dict)

    if not os.path.exists(data_dir + "/feature_add_s_nobin"):
        os.makedirs(data_dir + "/feature_add_s_nobin", exist_ok=True)

    with open(save_path, 'wb') as outfile:
        np.save(outfile, Feature)

def compute_feature_topo_compo(center_atom_vec, cart_enlarge_vec, pair_bettis):
    typ_dict = get_typ_dict(center_atom_vec)
    feature_topo = compute_feature_with_s_nobin(pair_bettis, typ_dict)
    feature_compo = compute_feature_composition(typ_dict)
    Feature = np.concatenate((feature_compo, feature_topo), axis=0)
    return Feature

def get_feature_topo_compo(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis, whole_bettis, alpha_bettis):
    name = id + "_feature.npy"
    save_path = data_dir + "/feature_topo_compo/" + name
    if os.path.exists(save_path):
        return

    Feature = compute_feature_topo_compo(center_atom_vec, cart_enlarge_vec, pair_bettis)

    if not os.path.exists(data_dir + "/feature_topo_compo"):
        os.makedirs(data_dir + "/feature_topo_compo", exist_ok=True)

    with open(save_path, "wb") as outfile:
        np.save(outfile, Feature)


def compute_feature_whole(center_atom_vec, cart_enlarge_vec, pair_bettis):
    pass

def get_feature_whole_compo(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis, whole_bettis, alpha_bettis):
    name = id + "_feature.npy"
    save_path = data_dir + "/feature_whole_compo/" + name
    if os.path.exists(save_path):
        return

    typ_dict = get_typ_dict(center_atom_vec)
    feature_compo = compute_feature_composition(typ_dict)
    feature_whole = compute_feature_whole(center_atom_vec, cart_enlarge_vec, pair_bettis)
    Feature = np.concatenate((feature_compo, feature_whole), axis=0)

    if not os.path.exists(data_dir + "/feature_whole_compo"):
        os.makedirs(data_dir + "/feature_whole_compo", exist_ok=True)

    with open(save_path, "wb") as outfile:
        np.save(outfile, Feature)