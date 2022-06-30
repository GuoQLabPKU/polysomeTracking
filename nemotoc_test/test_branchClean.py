import sys
sys.path.append('./')
import numpy as np
import pytest

from nemotoc_test.addRmPoly import setup, teardown
from nemotoc.polysome_class.polysome import Polysome

def swapIdx(idxPair):
    
    diff = idxPair[:,0] - idxPair[:,1]
    idx = np.where(diff>0)[0]
    if len(idx) == 0:
        return idxPair
    
    bp = idxPair[idx,0]
    idxPair[idx,0]=idxPair[idx,1]
    idxPair[idx,1] = bp
    
    return idxPair
    
    
    


def test_branchClean():   #maybe the best method is directly generate transList
    #generate polysome information
    #tomogram1
    conf = [ ]
    zz0 = { }
    zz0['type']='vect'
    zz0['tomoName']='100.mrc'
    zz0['numRepeats']=15
    zz0['increPos']=np.array([20, 40, 50])
    zz0['increAng']= np.array([30, 10, 70])
    zz0['startPos']=np.array([20, 30, 0])
    zz0['startAng']= np.array([40, 10, 30])
    zz0['minDist']=50
    zz0['searchRad']=100
    zz0['branch']=1
    zz0['noizeDregree'] = 0
    conf.append(zz0)
         
    #tomogram2
    zz3 = { }
    zz3['type']='vect'
    zz3['tomoName']='101.mrc'
    zz3['numRepeats']=25
    zz3['increPos']= np.array([60, 40, 10])
    zz3['increAng']= np.array([10, 20, 30])
    zz3['startPos'] = np.array([0, 0, 0])
    zz3['startAng']= np.array([50, 10, -30])
    zz3['minDist']=50
    zz3['searchRad']=100
    zz3['branch']=1
    zz3['noizeDregree'] = 0
    conf.append(zz3)
               
    idxBranches = setup(conf) #create simulation data  
    polysome1 = Polysome(input_star = './sim.star', run_time = 'run0')  
    polysome1.classify['clustThr'] = 5
    polysome1.sel[0]['minNumTransform'] = -1
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang

    polysome1.creatOutputFolder()
    
    polysome1.calcTransForms(worker_n = 1) #parallel, can assert the speed of pdit next time
   
    polysome1.groupTransForms(worker_n = 1) #parallel 
    
    _, _ = polysome1.selectTransFormClasses()  
                                 
    polysome1.alignTransforms()
    #find and delete branches
    polysome1.transForm['branchDepth'] = 0
    idxPair_rm = polysome1.find_connectedTransforms(None, 0) #0:don't save transList
    idxPair_rm = swapIdx(idxPair_rm)
    #compare the noisepair and idxpair_rm
    print('real dropped index pairs', idxBranches)
    print('code dropped index pairs')
    for i in range(idxPair_rm.shape[0]):
        print(idxPair_rm[i,:])
    for single_pairIdx in idxBranches:
        if (idxPair_rm == single_pairIdx).all(1).any():
            assert 1==1
        else:
            assert 1==0
        
if __name__ == '__main__':
    test_branchClean()
    teardown()
