import numpy as np
import pytest
from py_test.addRmPoly import setup
from py_test.addRmPoly import teardown
from py_io.tom_starread import tom_starread
from polysome_class.polysome import Polysome

import timeit as ti

def pick_polysome(input_star):
    polysome = dict()
  
    if isinstance(input_star, str):
        trans_star = tom_starread(input_star)
    else:
        trans_star = input_star
    unique_class = np.unique(trans_star['pairClass'].values)
    for single_class in unique_class:
        if single_class == 0:
            continue
        if single_class == -1:
            raise RuntimeError('No transform classes detected!')
        trans_star_singleclass = trans_star[trans_star['pairClass'] == single_class]
        pairlabel = np.unique(trans_star_singleclass['pairLabel'].values)
        for single_polysome in pairlabel:
            if single_polysome == -1.0:
                continue
            polysome_len = trans_star_singleclass[trans_star_singleclass['pairLabel'] == single_polysome].shape[0]
            if polysome_len >= 3:
                idx1 = trans_star_singleclass[trans_star_singleclass['pairLabel'] == single_polysome]['pairIDX1'].values
                idx2 = trans_star_singleclass[trans_star_singleclass['pairLabel'] == single_polysome]['pairIDX2'].values
                idx = np.unique(np.concatenate((idx1,idx2)))
                small_idx = np.min(idx)
                polysome[small_idx] = set(idx)
    return polysome


def test_polysome():
    #setup() 
    t1 = ti.default_timer()
    polysome1 = Polysome(input_star = './sim.star', run_time = 'run0')  
    polysome1.classify['clustThr'] = 5
    polysome1.classify['relinkWithoutSmallClasses'] = 0
    polysome1.sel[0]['minNumTransform'] = 0
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang
    #polysome1.fillPoly = np.array([1,4])
    polysome1.fillPoly = 'all'
    polysome1.creatOutputFolder()
    
    polysome1.calcTransForms(worker_n = 3) #parallel, can assert the speed of pdit next time
#    
    polysome1.groupTransForms(worker_n = 5) #parallel 
###                                         
    polysome1.alignTransforms()
###    
    polysome1.selectTransFormClasses()
###    
    polysome1.find_connectedTransforms()  #can assert here next time 
##    
    polysome1.analyseTransFromPopulation()
    
    polysome1.link_ShortPoly()
    
    polysome1.analyseTransFromPopulation()
    
    print('Finishing with %5.f seconds consumed.'%(ti.default_timer()-t1))
    #polysome1.visResult()
    track_polysome = pick_polysome('./cluster-sim/run0/allTransforms.star')   
    gen_polysome = np.load('./py_test/ori_polysome.npy',allow_pickle=True).item()
    print(track_polysome)
    print(gen_polysome)
    assert len(track_polysome) == len(gen_polysome)
#    for single_key in gen_polysome.keys():
#        assert gen_polysome[single_key] == track_polysome[single_key]
#        print('Tracked one right polysome!')
    #teardown()
    

if __name__ == '__main__':
    test_polysome()
                
                
        
        
        
        
        
        
        
