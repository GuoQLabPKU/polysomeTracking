import numpy as np
import os
import shutil
from py_test.genSimForwardPolyModel import genForwardPolyModel
from py_test.genSimNoise import genNoise

def setup(conf=None, noizeDegree = 2, branch = 0, eulerAngles = None, genType = 'poly'):
    '''
    Parameters
    ----------
    conf: dict with polysomes information stored. The default is None.

    Returns 
    -------
    simulated star file and store a npy file of polysomes

    '''
    #remove the previous sim.star and order random.star
    print('Generate simulation %s data.'%genType)
    if os.path.exists('sim.star'):
        os.remove('sim.star')
    if os.path.exists('simOrderRandomized.star'):
        os.remove('simOrderRandomized.star')
    if os.path.exists('sim_drop.star'):
        os.remove('sim_drop.star')
    
    if genType == 'poly':
        if conf == None:
            conf = [ ]
            zz0 = { }
            zz0['type']='vect'
            zz0['tomoName']='100.mrc'
            zz0['numRepeats']=15
            zz0['increPos']=np.array([20, 40, 50])
            zz0['increAng']= np.array([30, 10, 70])
            zz0['startPos']=np.array([20, 30, 0])
            zz0['startAng']= np.array([40, 10, 30])
            zz0['minDist']=50 #the pixel distance between two adajcent particles
            zz0['searchRad']=100
            if branch:
                zz0['branch']=1
            else:
                zz0['branch']=0
            zz0['noizeDregree'] = noizeDegree
            conf.append(zz0)
            
            zz1 = { }
            zz1['type']='vect'
            zz1['tomoName']='100.mrc'
            zz1['numRepeats']=30
            zz1['increPos']=np.array([60, 40, 10])
            zz1['increAng']= np.array([10, 20, 30])
            zz1['startPos']=np.array([500, 0, 0])
            zz1['startAng']= np.array([-20, -10, -30])
            zz1['minDist']=50
            zz1['searchRad']=100
            zz1['branch']=0
            zz1['noizeDregree'] = noizeDegree
            conf.append(zz1)
            
            zz2 = { }
            zz2['type']='noise'
            zz2['tomoName']='100.mrc'
            zz2['numRepeats']=250
            zz2['minDist']=50
            zz2['searchRad']=100
            conf.append(zz2)
            
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
            if branch:
                zz3['branch']=1
            else:
                zz3['branch']=0
            zz3['noizeDregree'] = noizeDegree
            conf.append(zz3)
            
            zz4 = { }
            zz4['type']='noise'
            zz4['tomoName']='101.mrc'
            zz4['numRepeats']=200
            zz4['minDist']=50
            zz4['searchRad']=100
            conf.append(zz4)
            idxBranches = genForwardPolyModel(conf, eulerAngles)
        else:
            idxBranches = genForwardPolyModel(conf, eulerAngles)
        return idxBranches
    else:
        genNoise(eulerAngles)
        
        
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
        
    if os.path.exists('sim_drop.star'):
        os.remove('sim_drop.star')
        
    if os.path.exists('sim_dropFillUp.star'):
        os.remove('sim_dropFillUp.star')
      
    if os.path.exists('simOrderRandomized.star'):
        os.remove('simOrderRandomized.star')
        
    if os.path.isdir('cluster-simOrderRandomized'):
        shutil.rmtree('cluster-simOrderRandomized')
       
    if os.path.isdir('cluster-sim_drop'):
        shutil.rmtree('cluster-sim_drop')
   
    if os.path.isdir('cluster-sim'):
        shutil.rmtree('cluster-sim')
    
    if os.path.isdir('cluster-simNoise'):
        shutil.rmtree('cluster-simNoise')
    
    if os.path.exists('simNoise.star'):
        shutil.rmtree('simNoise.star')
