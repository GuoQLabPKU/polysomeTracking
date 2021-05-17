import numpy as np
import os
from py_test.genForwardPolyModel import genForwardPolyModel
def setup(conf=None):
    '''
    Parameters
    ----------
    conf :dict, optional
        dict with polysomes information stored. The default is None.

    Returns 
    -------
    simulated star file and store a npy file of polysomes

    '''
    #remove the previous sim.star and order random.star
    print('Generating simulation data.')
    if os.path.exists('sim.star'):
        os.remove('sim.star')
    if os.path.exists('simOrderRandomized.star'):
        os.remove('simOrderRandomized.star')
    if conf == None:
        conf = [ ]
        zz0 = { }
        zz0['type']='vect'
        zz0['tomoName']='100.mrc'
        zz0['numRepeats']=15
        zz0['increPos']=np.array([60, 0, 50])
        zz0['increAng']= np.array([0, 0, 70])
        zz0['startPos']=np.array([500, 0, 0])
        zz0['startAng']= np.array([-20, -10, -30])
        zz0['minDist']=50
        zz0['searchRad']=100
        conf.append(zz0)
        
        zz1 = { }
        zz1['type']='vect'
        zz1['tomoName']='100.mrc'
        zz1['numRepeats']=30
        zz1['increPos']=np.array([60, 4, 10])
        zz1['increAng']= np.array([10, 20, 30])
        zz1['startPos']=np.array([500, 0, 0])
        zz1['startAng']= np.array([-20, -10, -30])
        zz1['minDist']=50
        zz1['searchRad']=100
        conf.append(zz1)
        
        zz2 = { }
        zz2['type']='noise'
        zz2['tomoName']='100.mrc'
        zz2['numRepeats']=750
        zz2['minDist']=50
        zz2['searchRad']=100
        conf.append(zz2)
        
        #tomogram2
        zz3 = { }
        zz3['type']='vect'
        zz3['tomoName']='101.mrc'
        zz3['numRepeats']=25
        zz3['increPos']= np.array([60, 40, 10])
        zz3['increAng']= np.array([0, 20, 30])
        zz3['startPos'] = np.array([0, 0, 0])
        zz3['startAng']= np.array([-20, -10, -30])
        zz3['minDist']=50
        zz3['searchRad']=100
        conf.append(zz3)
        
        zz4 = { }
        zz4['type']='noise'
        zz4['tomoName']='101.mrc'
        zz4['numRepeats']=750
        zz4['minDist']=50
        zz4['searchRad']=100
        conf.append(zz4)
            
        #tomogram3
        zz5 = { }
        zz5['type']='vect'
        zz5['tomoName']='102.mrc'
        zz5['numRepeats']=25
        zz5['increPos']= np.array([-5, 20, 10])
        zz5['increAng']= np.array([10, 0, 30])
        zz5['startPos'] = np.array([10, 10, 10])
        zz5['startAng']= np.array([0, -10, -30])
        zz5['minDist']=50
        zz5['searchRad']=100
        conf.append(zz5)
        
        zz6 = { }
        zz6['type']='noise'
        zz6['tomoName']='102.mrc'
        zz6['numRepeats']=850
        zz6['minDist']=50
        zz6['searchRad']=100
        conf.append(zz6)
        genForwardPolyModel(conf)
    else:
        genForwardPolyModel(conf)
        
def teardown():
    '''
    Returns
    -------
    None.

    '''
    #remove the previous sim.star and order random.star
    print('Remove simulation data.')
    if os.path.exists('sim.star'):
        os.remove('sim.star')
    if os.path.exists('simOrderRandomized.star'):
        os.remove('simOrderRandomized.star')
    if os.path.exists('cluster-simOrderRandomized/run0/allTransforms.star'):
        os.remove('cluster-simOrderRandomized/run0/allTransforms.star')
    if os.path.exists('cluster-simOrderRandomized/run0/scores/tree.npy'):
        os.remove('cluster-simOrderRandomized/run0/scores/tree.npy')
