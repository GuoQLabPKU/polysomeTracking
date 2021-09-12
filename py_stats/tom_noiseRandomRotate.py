import numpy as np
import random


from py_transform.tom_calcPairTransForm import tom_calcPairTransForm
from py_cluster.tom_A2Odist import tom_A2Odist



def tom_noiseRandomRotate(coord, eulerAngle, coordTarget, 
                  shift, rot, worker_n = 1, gpu_list = None, 
                  cmb_metric = 'scale2Ang', pruneRad = 100,
                  repeats = 500):

    #generate transList
    transListAct  =  np.array([]).reshape(-1, 6)
    for i in range(repeats):
        phi2, psi2, theta2 = random.uniform(-180,180), \
                        random.uniform(-180,180), random.uniform(-180,180)
        ang2 = np.array([phi2, psi2, theta2])
        posTr1, angTr1, _, _ = tom_calcPairTransForm(coord,
                                                    eulerAngle,
                                                    coordTarget,
                                                    ang2,'exact')
        
        transListAct = np.concatenate((transListAct,             
                             np.array([[posTr1[0], posTr1[1], posTr1[2], 
                                        angTr1[0], angTr1[1], angTr1[2],                                        
                                       ]])),axis = 0)
    
                       
    #check the distance between real data 
    eulerTarget = np.array([-61.152 , -49.5656, 101.8805])
    posTr1, angTr1, _, _ = tom_calcPairTransForm(coord,
                                                 eulerAngle,
                                                 coordTarget,
                                                 eulerTarget,'exact')
       
    transListAct = np.concatenate((transListAct,             
                             np.array([[posTr1[0], posTr1[1], posTr1[2], 
                                       angTr1[0], angTr1[1], angTr1[2],                                        
                                       ]])),axis = 0)
      
    #calculate distance with the mean transform
    _, distsAng, distsCN = tom_A2Odist(transListAct[:,0:3], 
                                             transListAct[:,3:6],
                                             shift,rot,
                                             worker_n, gpu_list, 
                                             cmb_metric, pruneRad)


    return distsAng[:-1], distsCN[:-1]
    
   
#if __name__ == '__main__':
#coord = np.array([1727.880,    1618.620,   649.036])
#_, eulerAngles = tom_eulerconvert_xmipp(145.482,  114.036,  -12.858)   
#coordTarget = np.array([1668.697,1662.067,637.071])
#_, eulerTarget = tom_eulerconvert_xmipp(151.15199, 101.880476, -40.43435)
#
#avgShift = np.array([-22.3, -56.4, 39.6])
#avgRot = np.array([-158.1, 156.7, 27.3])
#tom_noiseDist(coord, eulerAngles, coordTarget, 
#              avgShift, avgRot,saveDir = './',
#              worker_n = 1, gpu_list = None, 
#              cmb_metric = 'scale2Ang', pruneRad = 100,
#              randomRepeats = 1000 )
    