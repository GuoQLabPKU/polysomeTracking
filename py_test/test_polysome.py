import sys
import os
sys.path.append('/lustre/Data/jiangwh/polysome/python_version/polysome/')
import numpy as np
import pytest
from py_test.addRmPoly import setup, teardown
from py_io.tom_starread import tom_starread
from polysome_class.polysome import Polysome

####PARAMETERS#####
particleStar = './dataStar/allParticles_5goodtomos.star'
##################

def pick_polysome(input_star):
    polysome = dict()
  
    if isinstance(input_star, str):
        trans_star = tom_starread(input_star)
        trans_star = trans_star['data_particles']
    else:
        trans_star = input_star
    unique_class = np.unique(trans_star['pairClass'].values)
    for single_class in unique_class:
        if single_class == 0:
            continue
        if single_class == -1:
            raise RuntimeError('No transform classes detected!')
        transStar_singleclass = trans_star[trans_star['pairClass'] == single_class]
        pairlabel = np.unique(transStar_singleclass['pairLabel'].values)
        for single_polysome in pairlabel:
            if single_polysome == -1.0:
                continue
            polysome_len = transStar_singleclass[transStar_singleclass['pairLabel'] == single_polysome].shape[0]
            if polysome_len >= 3:
                idx1 = transStar_singleclass[transStar_singleclass['pairLabel'] == single_polysome] \
                                                                                    ['pairIDX1'].values
                idx2 = transStar_singleclass[transStar_singleclass['pairLabel'] == single_polysome] \
                                                                                    ['pairIDX2'].values
                idx = np.unique(np.concatenate((idx1,idx2)))
                small_idx = np.min(idx)
                polysome[small_idx] = set(idx)
    return polysome


def test_polysome(partStar):  
    if not os.path.exists(partStar):
        partStar = None
    _ = setup(realPartStar = partStar) #create simulation data  

    polysome1 = Polysome(input_star = './simOrderRandomized.star', run_time = 'run0')  
    polysome1.classify['clustThr'] = 5
    polysome1.classify['relinkWithoutSmallClasses'] = 0
    polysome1.sel[0]['minNumTransform'] = 10
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang

    polysome1.creatOutputFolder()
    
    polysome1.calcTransForms(worker_n = 3) #parallel, can assert the speed of pdit next time
   
    polysome1.groupTransForms(gpu_list = [0]) #parallel 
                                            
    polysome1.selectTransFormClasses()
    
    polysome1.alignTransforms()
    
    polysome1.find_connectedTransforms()  #can assert here next time 
    
    polysome1.analyseTransFromPopulation('',  '', 1)
    
    polysome1.noiseEstimate()
    
    #use advance mode
    polysome1.vis['vectField']['type'] = 'advance'
    polysome1.visResult()
    
    
      
    track_polysome = pick_polysome('./cluster-simOrderRandomized/run0/allTransforms.star')   
    real_polysome = np.load('./py_test/ori_polysome.npy',allow_pickle=True).item()
    assert len(track_polysome) == len(real_polysome)
    for single_key in real_polysome.keys():
        print('real poly idx:',real_polysome)
        print('tracking poly idx:',track_polysome)
        assert real_polysome[single_key] == track_polysome[single_key]
   

if __name__ == '__main__':
    test_polysome(particleStar)
    #teardown() #clean up the data 
                