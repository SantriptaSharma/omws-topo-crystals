from feature import *
from learning import *
from config import *
from structure import *
from multiprocessing import Pool

import numpy as np
import os

from timeit import default_timer as timer

def main():
    id_list = get_id_list(data_dir)

    # get feature
    if not SKIP_FEATURE:
        start = timer()
        if USE_MULTIPROCESS:
            pool = Pool(cpus)
            pool.map(batch_handle, split_list(id_list))
        else:
            batch_handle(id_list)
        end = timer()
        
        print(f"Joined in {end - start} seconds")
 
    learning_cv(data_dir)

if __name__ == '__main__':
    main()