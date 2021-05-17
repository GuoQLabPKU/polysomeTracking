import numpy as np
from py_cluster.tom_pdist import calcRotMatrices
from py_cluster.tom_pdist import calcAngDist
from scipy.linalg import blas as FB
import cupy as cp

import timeit as ti

def returnMatrix():
    #only test the calculation of angle distance of rotation matrixs
    #load the in_FW ans in_Inv 
    in_Fw = np.load('./py_test/test_pdist/in_Fw.npy',allow_pickle=True) 
    in_Inv = np.load('./py_test/test_pdist/in_Inv.npy', allow_pickle = True)
    Rin = calcRotMatrices(in_Fw)
    Rin_Inv = calcRotMatrices(in_Inv)
    jobListChunk = np.load('./py_test/test_pdist/jobListChunk_0.npy',allow_pickle=True)
    Rs = Rin[jobListChunk[:,0],:,0:3]
    RsInv = Rin[jobListChunk[:,1],:,3:6]
    Rs_Inv = Rin_Inv[jobListChunk[:,0],:,0:3]
    Rs_Inv_Inv = Rin_Inv[jobListChunk[:,1],:,3:6]
    return Rs, RsInv, Rs_Inv, Rs_Inv_Inv
    
#def test_pdistori():
#    '''
#    original pdist method
#    '''
#    start = ti.default_timer()
#    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
#    dtmp = calcAngDist(Rs, RsInv)
#    dtmpInv = calcAngDist(Rs_Inv, RsInv)
#    dtmpInv2 = calcAngDist(Rs, Rs_Inv_Inv)
#    dtmpInv3 = calcAngDist(Rs_Inv, Rs_Inv_Inv )
#    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
#    dists = np.min(dists_all, axis = 0).astype(np.single)   
#    #load the saved dists 
#    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
#    assert np.allclose(dists, expected, rtol=1e-02), (dists, expected)
#    end = ti.default_timer()    
#    print('done. pdistori consumed %.5f secs'%(end-start))
        


def calcAngDist_matmul(Rs,RsInv):
    #multiple the two matrices arrays
    t1 = ti.default_timer()
    Rp = np.matmul(Rs, RsInv)
    print('multiply',ti.default_timer() - t1)
    t1 = ti.default_timer()
    tr_Rp = (np.trace(Rp, axis1=1, axis2=2) - 1)/2
    print('trace',ti.default_timer() - t1)
    #calculate the angle distance 
    t1 = ti.default_timer()
    dists = np.lib.scimath.arccos(tr_Rp)/np.pi*180 
    print('arccos',ti.default_timer() - t1)
    #extract the real part of the dists and single them
    return dists.real #one dimention arrsy float32)

def test_pdistmatmul():
    '''
    using matmul broadcast
    '''
    start = ti.default_timer()
    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
    dtmp = calcAngDist_matmul(Rs, RsInv)
    dtmpInv = calcAngDist_matmul(Rs_Inv, RsInv)
    dtmpInv2 = calcAngDist_matmul(Rs, Rs_Inv_Inv)
    dtmpInv3 = calcAngDist_matmul(Rs_Inv, Rs_Inv_Inv )
    t1 = ti.default_timer()
    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
    print('combination four distances',ti.default_timer() - t1)
    t1 = ti.default_timer()
    dists = np.min(dists_all, axis = 0).astype(np.single) 
    print('min',ti.default_timer() - t1)
    #load the saved dists 
    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
    assert np.allclose(dists, expected, rtol=1e-02), (dists, expected) 
    end = ti.default_timer()
    print('done. pdistmatmul consumed %.5f secs'%(end-start))


#def calcAngDist_blas(Rs,RsInv):
#    tr_Rp = np.zeros(Rs.shape[0], dtype = np.single)
#    for i in range(Rs.shape[0]):
#        Rp = FB.sgemm(1.0, Rs[i], RsInv[i], True)
#        tr_Rp[i] = np.trace(Rp)
#    #calculate the angle distance 
#    dists = np.array([np.lib.scimath.arccos(i)/np.pi*180 for i in (tr_Rp-1)/2])
#    #extract the real part of the dists and single them
#    dists = np.single(dists.real)
#    
#    return dists #one dimention arrsy float32
#    
#    
#def test_pdistblast():
#    '''
#    using blast
#    '''
#    start = ti.default_timer()
#    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
#    #change the data rank
#    Rs = np.array(Rs, order = 'F')
#    RsInv = np.array(RsInv, order = 'F')
#    Rs_Inv = np.array(Rs_Inv, order = 'F')
#    Rs_Inv_Inv = np.array(Rs_Inv_Inv, order = 'F')
#    
#    dtmp = calcAngDist_blas(Rs, RsInv)
#    dtmpInv = calcAngDist_blas(Rs_Inv, RsInv)
#    dtmpInv2 = calcAngDist_blas(Rs, Rs_Inv_Inv)
#    dtmpInv3 = calcAngDist_blas(Rs_Inv, Rs_Inv_Inv )
#    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
#    dists = np.min(dists_all, axis = 0).astype(np.single)   
#    #load the saved dists 
#    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
#    assert np.allclose(dists, expected, rtol=1e-02), (dists, expected) 
#    end = ti.default_timer()
#    print('done. pdistblast consumed %.5f secs'%(end-start))
 
    
#def calcAngDist_matmulsingle(Rs,RsInv):
#    tr_Rp = np.zeros(Rs.shape[0], dtype = np.single)
#    for i in range(Rs.shape[0]):
#        Rp = np.matmul(Rs[i], RsInv[i])
#        tr_Rp[i] = np.trace(Rp)
#    #calculate the angle distance 
#    dists = np.array([np.lib.scimath.arccos(i)/np.pi*180 for i in (tr_Rp-1)/2])
#    #extract the real part of the dists and single them
#    dists = np.single(dists.real)
#    
#    return dists #one dimention arrsy float32
#    
#    
#def test_pdistmatmulsingle():
#    '''
#    using blast
#    '''
#    start = ti.default_timer()
#    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
#    #change the data rank
#   
#    dtmp = calcAngDist_matmulsingle(Rs, RsInv)
#    dtmpInv = calcAngDist_matmulsingle(Rs_Inv, RsInv)
#    dtmpInv2 = calcAngDist_matmulsingle(Rs, Rs_Inv_Inv)
#    dtmpInv3 = calcAngDist_matmulsingle(Rs_Inv, Rs_Inv_Inv )
#    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
#    dists = np.min(dists_all, axis = 0).astype(np.single)   
#    #load the saved dists 
#    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
#    assert np.allclose(dists, expected, rtol=1e-02), (dists, expected) 
#    end = ti.default_timer()
#    print('done. pdistblast consumed %.5f secs'%(end-start))
 

#def calcAngDist_multiply(Rs,RsInv):
#    #multiple the two matrices arrays
#    Rp = np.multiply(Rs, RsInv)
#    tr_Rp = np.array([np.trace(Rp[i]) for i in range(Rp.shape[0])])
#    #calculate the angle distance 
#    dists = np.array([np.lib.scimath.arccos(i)/np.pi*180 for i in (tr_Rp-1)/2])
#    #extract the real part of the dists and single them
#    dists = np.single(dists.real)
#    
#    return dists #one dimention arrsy float32)
#    
#def test_multiply():
#    '''
#    using blast
#    '''
#    start = ti.default_timer()
#    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
#    #change the data rank
#   
#    dtmp = calcAngDist_multiply(Rs, RsInv)
#    dtmpInv = calcAngDist_multiply(Rs_Inv, RsInv)
#    dtmpInv2 = calcAngDist_multiply(Rs, Rs_Inv_Inv)
#    dtmpInv3 = calcAngDist_multiply(Rs_Inv, Rs_Inv_Inv )
#    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
#    dists = np.min(dists_all, axis = 0).astype(np.single)   
#    #load the saved dists 
#    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
#    assert np.allclose(dists, expected, rtol=1e-02), (dists, expected) 
#    end = ti.default_timer()
#    print('done. pdistblast consumed %.5f secs'%(end-start)) 


def cupy_multiply(Rs,RsInv):
    t1 = ti.default_timer()
    Rs_cu = cp.asarray(Rs)
    RsInv_cu = cp.asarray(RsInv)
    print('from numpy to cupy',ti.default_timer() - t1)
    t1 = ti.default_timer()
    Rp = cp.matmul(Rs_cu,RsInv_cu)
    print('multiply',ti.default_timer() - t1)
    t1 = ti.default_timer()
    tr_Rp = (cp.trace(Rp, axis1=1, axis2=2) - 1)/2
    print('trace',ti.default_timer() - t1)
    t1 = ti.default_timer()
    tr_Rp = cp.clip(tr_Rp, a_min = -1, a_max =1)
    dists = cp.arccos(tr_Rp)/np.pi*180
    print('arccos',ti.default_timer() - t1)
    t1 = ti.default_timer()
    dists = cp.asnumpy(dists.real)
    print('from cupy to numpy',ti.default_timer() - t1)
    return dists

    
def test_cupy():
    '''
    using cupy
    '''
    start = ti.default_timer()
    Rs, RsInv, Rs_Inv, Rs_Inv_Inv = returnMatrix()
    #change the data rank 
    dtmp = cupy_multiply(Rs, RsInv)
    dtmpInv = cupy_multiply(Rs_Inv, RsInv)
    dtmpInv2 = cupy_multiply(Rs, Rs_Inv_Inv)
    dtmpInv3 = cupy_multiply(Rs_Inv, Rs_Inv_Inv )
    t1 = ti.default_timer()    
    dists_all = np.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
    print('combination four distances',ti.default_timer() - t1)
    t1 = ti.default_timer()
    dists = np.min(dists_all, axis = 0).astype(np.single)   
    #load the saved dists 
    print('min %.5f'%(ti.default_timer()-t1))
    expected = np.load('./py_test/test_pdist/dist.npy', allow_pickle=True)
    assert np.allclose(dists, expected, rtol=1e-01), (dists, expected) 
    end = ti.default_timer()
    print('done. pdistblast consumed %.5f secs'%(end-start))    
