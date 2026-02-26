import os
import numpy as np
from config import *
from ripser import Rips, ripser
import pickle as pkl

from feature import element_properties

from gudhi import AlphaComplex

et = element_properties.index.to_numpy(dtype="S2")
er = element_properties["CovalentRadius"].to_numpy()

# poscar to numpy array
def get_prim_structure_info(data_dir, id):
    save_path = data_dir + '/atoms/' + id + '_original.npz'
    if os.path.exists(save_path):
        return

    # need to change if predict icsd data
    with open(data_dir + '/structure/' + id, 'r') as f:
        lines = f.read().splitlines()
    atom_map = []
    index_up = 0
    for atom, num in zip(lines[5].split(), lines[6].split()):
        index_up += int(num)
        atom_map.append((atom, index_up))
    lattice_vec = []
    for line in lines[2:5]:
        x, y, z = line.split()
        lattice_vec.append([float(x), float(y), float(z)])
    lattice_vec = np.array(lattice_vec)
    #get atom position
    index_atom = 0
    atom_nums = len(lines[8:])
    atom_vec = np.zeros([atom_nums], dtype=dt)
    for i in range(atom_nums):
        line = lines[8+i]
        x, y, z = line.split()
        if i < atom_map[index_atom][1]:
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        else:
            index_atom += 1
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        atom_vec[i]['pos'][:] = np.array([float(x), float(y), float(z)])
    
    if not os.path.exists(data_dir + "/atoms"):
        os.makedirs(data_dir + "/atoms", exist_ok=True)

    with open(save_path, 'wb') as out_file:
        np.savez(out_file, lattice_vec=lattice_vec, atom_vec=atom_vec)


# enlarge the unit cell to each atom in unit cell can form a ball with radius value cut
def enlarge_cell(data_dir, id):
    save_path = data_dir + '/atoms/' + id + '_enlarge.npz'
    if os.path.exists(save_path):
        with open(save_path, 'rb') as structfile:
            data = np.load(structfile)
            return data['CAV'], data['CEV']

    with open(data_dir + '/atoms/' + id + '_original.npz', 'rb') as structfile:
        data = np.load(structfile)
        lattice_vec = data['lattice_vec']; atom_vec = data['atom_vec']
    min_lattice = min([np.linalg.norm(i) for i in lattice_vec])
    mul_time = int(np.ceil(cut/min_lattice))
    center_atom_vec = atom_vec.copy()
    center_atom_vec['pos'][:] += mul_time
    center_atom_vec['pos'][:] = np.matmul(center_atom_vec['pos'][:], lattice_vec)

    atom_nums = (mul_time*2 + 2)**3 * len(atom_vec)
    enlarge_vec = np.zeros([atom_nums], dtype=dt)
    cart_enlarge_vec = np.zeros([atom_nums], dtype=dt)
    i = 0

    for atom in atom_vec:
        typ = atom['typ']

        xv, yv, zv = np.meshgrid(np.arange(mul_time*2 + 2), np.arange(mul_time*2 + 2), np.arange(mul_time*2 + 2))
        positions = np.repeat(atom['pos'][None, :], xv.size, axis=0) + np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)

        enlarge_vec[i: i + xv.size]['typ'] = typ
        enlarge_vec[i: i + xv.size]['pos'] = positions
        i += xv.size

    cart_enlarge_vec[:]['typ'] = enlarge_vec[:]['typ']    
    cart_enlarge_vec[:]['pos'] = np.matmul(enlarge_vec[:]['pos'], lattice_vec)

    with open(save_path, 'wb') as out_file:
        np.savez(out_file, CAV=center_atom_vec, CEV=cart_enlarge_vec)

    return center_atom_vec, cart_enlarge_vec


# betti number for one structure
def get_betti_num(data_dir, id, center_atom_vec, cart_enlarge_vec):
    save_path = data_dir + '/betti_num/' + id + ".pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as phfile:
            all_pair_outs = pkl.load(phfile)
            return all_pair_outs
 
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ']
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1
    
    # for every atom in center_atom
    # first calculate it neighbor atom with same element
    # then with other element within distance cut
    if not os.path.exists(data_dir + "/betti_num"):
        os.makedirs(data_dir + "/betti_num", exist_ok=True)

    all_pair_outs = []
    for cav in center_atom_vec:
        center_type = cav['typ']

        for ele in typ_dict.keys():
            dist = np.linalg.norm(cart_enlarge_vec[:]['pos'] - cav['pos'], axis=1)
            included_points = cart_enlarge_vec[((cart_enlarge_vec[:]['typ'] == ele) | (cart_enlarge_vec[:]['typ'] == center_type)) & (dist <= cut)]

            points_num = len(included_points)
            if points_num == 0:
                continue

            points = np.zeros((points_num+1, 3))
            points[0][:] = cav['pos'][:]
            points[1:][:] = included_points[:]['pos'][:]

            dgms = ripser(points, maxdim=2, thresh=cut)['dgms']
            all_pair_outs.append((center_type.decode(), ele.decode(), dgms))

    with open(save_path, 'wb') as out_file:
        pkl.dump(all_pair_outs, out_file)

    return all_pair_outs

def get_betti_whole_lattice(data_dir, id, cav, cev):
    save_path = data_dir + '/betti_num/' + id + "_whole.pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as phfile:
            whole_lattice_out = pkl.load(phfile)
            return whole_lattice_out

    points = cev['pos'][:]
    dgms = ripser(points, maxdim=2, thresh=cut)['dgms']

    with open(save_path, 'wb') as out_file:
        pkl.dump(dgms, out_file)

    return dgms

def get_betti_weighted_alpha(data_dir, id, cav, cev):
    save_path = data_dir + '/betti_num/' + id + "_walpha.pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as phfile:
            walpha_out = pkl.load(phfile)
            return walpha_out

    points = cev['pos'][:]
    weights = er[np.where(cev['typ'][:, None] == et[None, :])[1]]

    assert points.shape[0] == weights.shape[0] and points.shape[1] == 3

    st = AlphaComplex(points=points, weights=weights).create_simplex_tree()
    bars_non_graded = st.persistence(homology_coeff_field=2, persistence_dim_max=True)

    bars = [[] for _ in range(3)]
    for dim, (birth, death) in bars_non_graded:
        bars[dim].append((birth, death))

    with open(save_path, 'wb') as out_file:
        pkl.dump(bars, out_file)

    return bars