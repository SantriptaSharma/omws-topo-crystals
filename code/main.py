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
            args = [(i, id_list[i::cpus]) for i in range(cpus)]
            pool.starmap(batch_handle, args)
        else:
            batch_handle(0, id_list)
        end = timer()
        
        print(f"Joined in {end - start} seconds")
 
    if not SKIP_LEARNING:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"worker_logs/learning.log", filemode='a')
        learning_cv(data_dir)

if __name__ == '__main__':
    main()