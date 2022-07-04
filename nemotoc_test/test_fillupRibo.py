import sys
sys.path.append('./')
import numpy as np
import pytest


from nemotoc_test.addRmPoly import setup, teardown
from nemotoc.py_io.tom_starread import tom_starread
from nemotoc.py_io.tom_starwrite import tom_starwrite
from nemotoc.polysome_class.polysome import Polysome


def generateDeletPoly():
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
    zz0['branch']=0
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
    zz3['branch']=0
    zz3['noizeDregree'] = 0
    conf.append(zz3)  
    
    _ = setup(conf)
    simStar = tom_starread('./sim.star')
    simStar_ = simStar['data_particles']
    drop_index = [25,26]
    ribo_info = simStar_.iloc[drop_index,:]       
    simStar_.drop(index = drop_index,inplace = True) 
    simStar['data_particles'] = simStar_
    tom_starwrite('./sim_drop.star', simStar)
    
    return ribo_info.loc[:, ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].values




def test_branchClean():   
    riboDrop_info = generateDeletPoly()#create simulation data  
    polysome1 = Polysome(input_star = './sim_drop.star', run_time = 'run0') 
    polysome1.classify['clustThr'] = 5
    polysome1.sel[0]['minNumTransform'] = -1
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang

    polysome1.creatOutputFolder()
    
    polysome1.calcTransForms(worker_n = 1) #parallel, can assert the speed of pdit next time
   
    polysome1.groupTransForms(worker_n = 1) #parallel 
                                         
    polysome1.alignTransforms()
    
    polysome1.find_connectedTransforms()  
    
    #vis that the polysomes with missing ribosomes
#    vectVisP = polysome1.vis['vectField']
#    tom_plot_vectorField(polysome1.transList, vectVisP['showTomo'], 
#                                  vectVisP['showClass'], vectVisP['polyNr'], vectVisP['onlySelected'],
#                                  vectVisP['repVectLen'],vectVisP['repVect'],
#                                  np.array([0,0,1]), '', '')
     
    polysome1.analyseTransFromPopulation()
    
    #link the polysomes
    fillPoly = { }
    fillPoly['addNum'] = 2
    fillPoly['fitModel'] = 'lognorm' 
    fillPoly['threshold'] = 0.05
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly()
    
#    tom_plot_vectorField(polysome1.transList, vectVisP['showTomo'], 
#                                  vectVisP['showClass'], vectVisP['polyNr'], vectVisP['onlySelected'],
#                                  vectVisP['repVectLen'],vectVisP['repVect'],
#                                  np.array([0,0,1]), '', '')   
    #load the filled up particle.star and get the filled up ribosomes
    particlesData = tom_starread('./cluster-sim_drop/run0/sim_dropFillUp.star')
    particlesData = particlesData['data_particles']
    fillupRibos = particlesData[particlesData['if_fillUp'] != -1]
    fillupRibos_coord = fillupRibos.loc[:,['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].values
    print('real drop:', riboDrop_info)
    print('code filled up:', fillupRibos_coord)
    assert riboDrop_info.shape[0] == fillupRibos_coord.shape[0]
    
    #get the connection between real/code drop ribosomes
    if np.linalg.norm(fillupRibos_coord[0]-riboDrop_info[0]) < 10e-3:
        assert np.linalg.norm(fillupRibos_coord[1]-riboDrop_info[1])
    else:
        assert np.linalg.norm(fillupRibos_coord[0]-riboDrop_info[1]) < 10e-3
        assert np.linalg.norm(fillupRibos_coord[1]-riboDrop_info[0]) < 10e-3
        
     
if __name__ == '__main__':
    test_branchClean()
    teardown()

