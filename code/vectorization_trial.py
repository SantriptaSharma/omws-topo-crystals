# vectorization trial code
from ripser import Rips, ripser
import pickle as pkl
import os
import numpy as np
import vectorization as vec

data_dir = "../datalk/datatda"
cut = 0.8
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

def get_betti_num_vec(data_dir, id, center_atom_vec, cart_enlarge_vec):
    save_path = data_dir + '/betti_num_vec/' + id + ".pkl"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as phfile:
            all_pair_outs, feature_vec = pkl.load(phfile)
            return all_pair_outs, feature_vec
 
    typ_dict = {}
    for vect in center_atom_vec:
        typ = vect['typ']
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1
    
    # for every atom in center_atom
    # first calculate it neighbor atom with same element
    # then with other element within distance cut
    if not os.path.exists(data_dir + "/betti_num_vec"):
        os.makedirs(data_dir + "/betti_num_vec", exist_ok=True)

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
            feature_vec = []
            for i in range(3):
                tmp = dgms[i]
                # l = np.array([p[1] - p[0] for p in tmp])
                feature_vec.append(vec.GetBettiCurveFeature(np.array(tmp)))
            all_pair_outs.append((center_type.decode(), ele.decode(), dgms))

    with open(save_path, 'wb') as out_file:
        pkl.dump((all_pair_outs, feature_vec), out_file)

    return (all_pair_outs, feature_vec)

with open(data_dir + '/properties.txt', 'r') as f:
    lines = f.read().splitlines()
    id_list = []
    for line in lines[1:]:
        id = line.split()[0]
        id_list.append(id)
for id in id_list:
    cav, cev = enlarge_cell(data_dir, id)
    (all_pair_outs, feature_vec) = get_betti_num_vec(data_dir, id, cav, cev)
    # print(feature_vec)


