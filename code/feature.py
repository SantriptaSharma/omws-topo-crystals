import numpy as np
import pandas as pd
import os
import math
import re
from config import *

import pickle as pkl

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

def get_feature_composition(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis):
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
    bar_ctypes = [[] for _ in range(3)]
    bar_births = [[] for _ in range(3)]
    bar_deaths = [[] for _ in range(3)]

    bar_births_weighted = [[] for _ in range(3)]
    bar_deaths_weighted = [[] for _ in range(3)]

    Feature_3 = np.zeros([com_len, 7*5], float)

    for (ctype, eltype, dgms) in pair_bettis:
        ca_num = typ_dict[ctype]
        cp_idx = common_pair[ctype.decode()]
    
        for i, dgm in enumerate(dgms):
            dim = i
            
            bar_ctypes[dim].extend([ctype] * len(dgm))
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



    bars_ctypes = [np.array(bar_ctypes[i]) for i in range(3)]
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


    for (ctype, eltype, dgms) in pair_bettis:
        if ctype not in common_pair:
            continue
            
        ca_num = typ_dict[ctype]
        idx = common_pair[ctype]
        
        for dim, dgm in enumerate(dgms):
            if len(dgm) == 0:
                continue
                
            births = np.array([birth for birth, death in dgm])
            deaths = np.array([death if death != float('inf') else -1 for birth, death in dgm])
            valid_mask = deaths != -1
            valid_deaths = deaths[valid_mask]
            


            if dim == 0 and len(valid_deaths) > 0:
                Feature_3[idx, 0] = np.mean(valid_deaths)
                Feature_3[idx, 1] = np.std(valid_deaths)
                Feature_3[idx, 2] = np.max(valid_deaths)
                Feature_3[idx, 3] = np.min(valid_deaths)
                Feature_3[idx, 4] = np.sum(valid_deaths) / ca_num
                
            elif dim == 1 and len(valid_deaths) > 0:
                valid_births = births[valid_mask]
                lifetimes = valid_deaths - valid_births
                
                Feature_3[idx, 5] = np.mean(lifetimes)
                Feature_3[idx, 6] = np.std(lifetimes)
                Feature_3[idx, 7] = np.max(lifetimes)
                Feature_3[idx, 8] = np.min(lifetimes)
                Feature_3[idx, 9] = np.sum(lifetimes) / ca_num
                Feature_3[idx, 10] = np.mean(valid_births)
                Feature_3[idx, 11] = np.std(valid_births)
                Feature_3[idx, 12] = np.max(valid_births)
                Feature_3[idx, 13] = np.min(valid_births)
                Feature_3[idx, 14] = np.sum(valid_births) / ca_num
                Feature_3[idx, 15] = np.mean(valid_deaths)
                Feature_3[idx, 16] = np.std(valid_deaths)
                Feature_3[idx, 17] = np.max(valid_deaths)
                Feature_3[idx, 18] = np.min(valid_deaths)
                Feature_3[idx, 19] = np.sum(valid_deaths) / ca_num
                
            elif dim == 2 and len(valid_deaths) > 0:
                valid_births = births[valid_mask]
                lifetimes = valid_deaths - valid_births
                
                Feature_3[idx, 20] = np.mean(lifetimes)
                Feature_3[idx, 21] = np.std(lifetimes)
                Feature_3[idx, 22] = np.max(lifetimes)
                Feature_3[idx, 23] = np.min(lifetimes)
                Feature_3[idx, 24] = np.sum(lifetimes) / ca_num
                Feature_3[idx, 25] = np.mean(valid_births)
                Feature_3[idx, 26] = np.std(valid_births)
                Feature_3[idx, 27] = np.max(valid_births)
                Feature_3[idx, 28] = np.min(valid_births)
                Feature_3[idx, 29] = np.sum(valid_births) / ca_num
                Feature_3[idx, 30] = np.mean(valid_deaths)
                Feature_3[idx, 31] = np.std(valid_deaths)
                Feature_3[idx, 32] = np.max(valid_deaths)
                Feature_3[idx, 33] = np.min(valid_deaths)
                Feature_3[idx, 34] = np.sum(valid_deaths) / ca_num

    Feature_3 = np.concatenate(Feature_3, axis=0)
    Feature = np.concatenate((Feature_2, Feature_3), axis=0)
    return Feature

def get_feature_with_s_nobin(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis):
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

def get_feature_topo_compo(data_dir, id, center_atom_vec, cart_enlarge_vec, pair_bettis):
    name = id + "_feature.npy"
    save_path = data_dir + "/feature_topo_compo/" + name
    if os.path.exists(save_path):
        return

    Feature = compute_feature_topo_compo(center_atom_vec, cart_enlarge_vec, pair_bettis)

    if not os.path.exists(data_dir + "/feature_topo_compo"):
        os.makedirs(data_dir + "/feature_topo_compo", exist_ok=True)

    with open(save_path, "wb") as outfile:
        np.save(outfile, Feature)

def atoms_frequency():
    sub_dict = split_element(data_dir)

    atom_single = {}
    for sub in sub_dict.keys():
        for i in sub_dict[sub]:
            if i not in atom_single:
                atom_single[i] = 0.0
            atom_single[i] += 1.0
    return atom_single

def split_element(data_dir):
    pattern = re.compile("[A-Z]{1}[a-z]{0,1}")
    # substance to all pair
    sub_dict = {}

    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        sub = line.split()[0]
        with open(data_dir + "/structure/" + sub, 'r') as tmp:
            ls = tmp.read().splitlines()[0]
        eles = set(pattern.findall(ls))
        sub_dict[sub] = eles
    return sub_dict