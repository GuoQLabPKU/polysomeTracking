# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:49:50 2021

@author: DELL
"""
from multiprocessing import Process
import multiprocessing as mp
import os
import numpy as np
import gc
import time

def level1():
    a,b = level2()
    print(a)
    print(b)

def level2():
    a,b = level3()
    return a,b
def level3():
    data_to_process = np.random.rand(1000000,4)
#    scores = np.zeros([1000000,3])
#    for single_row in range(data_to_process.shape[0]):
#        a,b,c,_ = data_to_process[single_row,:]
#        scores[single_row,:] = np.array([a,b,a+b+c])
    scores = level4(data_to_process)
    return data_to_process,scores
def level4(data_input):
    npr = 1
    processes = [ ]
    n_part = data_input.shape[0]
    sym_ids = np.arange(n_part)
    spl_ids = np.array_split(range(len(sym_ids)),npr)
    shared_ncc = mp.Array('f',n_part*3)
    for pr_id in range(npr):
        read_row = list(spl_ids[pr_id])
        data_toprocess = data_input[read_row,:]
        pr = mp.Process(target=level5_t, args = (pr_id, read_row, data_toprocess, shared_ncc))
        pr.start()
        processes.append(pr)
    pr_results = [ ]
    for pr in processes:
        pr.join()
        pr_results.append(pr.exitcode)
    for pr_id in range(len(processes)):
        if pr_id != pr_results[pr_id]:
            print("warning")
    
    gc.collect()  #delete the garbage
    scores = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(n_part,3)
    return scores
def level5_t(pr_id,rows_id, data_toprocess,shared_ncc):
    for single_row,single_read_row in zip(range(data_toprocess.shape[0]), rows_id):
        a,b,c,_ = data_toprocess[single_row,:]
        if single_read_row == 0.0:
            shared_ncc[0] = a; shared_ncc[1] = b; shared_ncc[2] = a+b+c
        else:
            shared_ncc[single_read_row*3] = a;shared_ncc[single_read_row*3+1] = b;shared_ncc[single_read_row*3+2] = a+b+c

    os._exit(pr_id)    

if __name__ == '__main__':
    t1 = time.time()
    level1()
    t2 = time.time()
    print(t2-t1)
    