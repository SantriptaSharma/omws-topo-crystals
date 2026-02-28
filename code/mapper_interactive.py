from tqdm import tqdm

import pickle as pkl
import pandas as pd
import numpy as np
import os

data_dir = "data/"

datapoints = [b.split("_feature.npy")[0] for b in os.listdir("data/feature_topo_compo")]

feat_topo_compo_paths = os.listdir("data/feature_topo_compo")
feat = []
for item in tqdm(feat_topo_compo_paths, desc="Loading features", total=len(feat_topo_compo_paths)):
    feat.append(np.load("data/feature_topo_compo/" + item))
feat_new = np.array(feat)[:, 160:]

with open(data_dir + '/properties.txt', 'r') as f:
    lines = f.read().splitlines()
    id_list_old = []
    formation_enthalpy_old = []
    for line in tqdm(lines[1:], desc="Processing properties.txt", total=len(lines)-1):
        id = line.split()[0]
        form_enth = line.split()[1]
        # if np.any(np.char.find(feat_topo_compo_paths, id)) >= 0:
        formation_enthalpy_old.append(form_enth) 
        id_list_old.append(id)

print(len(id_list_old), feat_new.shape[1], len(formation_enthalpy_old))
# indicies = np.array([np.any(np.char.startswith(feat, char)) for char in id_list_old]
df_1 = pd.DataFrame({'id' : id_list_old, 'formation_enthalpy' : formation_enthalpy_old})
df_1 = pd.concat([df_1, pd.DataFrame(feat_new)], axis=1)

df_1.to_csv('mapper_data.csv', index=False)

