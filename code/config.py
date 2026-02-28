import numpy as np

small = 0.0001
cut = 8.0
rs = 0.25
cpus = 12
# change it to your data directory
data_dir = "./data"
dt = np.dtype([('typ', 'S2'), ('pos', float, (3, ))])
lth = int(np.rint(cut/rs))
# feature name
fname = "feature_alpha_compo"

USE_MULTIPROCESS = True
SKIP_FEATURE = True
SKIP_LEARNING = False

n_estimators_arr = [500, 5000]
BETTI_CURVE_SAMPLES = 100