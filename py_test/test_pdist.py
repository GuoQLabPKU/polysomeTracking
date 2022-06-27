import sys
sys.path.append('./')
import numpy as np
import pytest
from py_cluster.tom_pdist_cpu import tom_pdist as tp1
from py_cluster.tom_pdist_gpu import tom_pdist as tp2

def eulerInv(ang):
    angInv =  np.array([-ang[1], -ang[0], -ang[2]])   
    return angInv
    
    
    
if __name__ == '__main__':  
    #check cpu version
    angFw = np.array([[0,0,30], [0,0,30]])
    angInv = np.array([[0,0,-30],[0,0,-30]])   
    d1_ang = tp1(angFw,1000,1,None,'ang',angInv)
    
    vectFw = np.array([[10,20,30], [10,20,30]])
    vectInv = np.array([[-10,-20,-30], [-10,-20,-30]])
    d1_vect = tp1(vectFw,1000,1,None,'euc',vectInv)
    
    #check the gpu version 
    maxchunk = {0:1000}
    d2_ang = tp2(angFw,maxchunk,1,None,'ang',angInv)
    d2_vect = tp2(vectFw,maxchunk,1,None,'euc',vectInv)
    
    assert len(d1_ang) == 1
    assert d1_ang[0] < 10e-5
    
    assert len(d2_ang) == 1
    assert d2_ang[0] < 10e-5
    
    assert len(d1_vect) == 1
    assert d1_vect[0] < 10e-5
    
    assert len(d2_vect) == 1
    assert d2_vect[0] < 10e-5
