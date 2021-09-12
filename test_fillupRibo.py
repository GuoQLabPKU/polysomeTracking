import numpy as np
import pytest
import timeit as ti

from py_test.addRmPoly import setup, teardown
from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from polysome_class.polysome import Polysome
from py_vis.tom_plot_vectorField import tom_plot_vectorField

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
    drop_index = [25,26]
    ribo_info = simStar.iloc[drop_index,:]       
    simStar.drop(index = drop_index,inplace = True)
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"]  = ["_%s"%i for i in simStar.columns]
    tom_starwrite('./sim_drop.star', simStar, header)
    
    return ribo_info.loc[:, ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].values




def test_branchClean():   
    riboDrop_info = generateDeletPoly()#create simulation data  
    polysome1 = Polysome(input_star = './sim_drop.star', run_time = 'run0') 
    polysome1.classify['clustThr'] = 5
    polysome1.classify['relinkWithoutSmallClasses'] = 1
    polysome1.sel[0]['minNumTransform'] = 20
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang

    polysome1.creatOutputFolder()
    
    polysome1.calcTransForms(worker_n = 3) #parallel, can assert the speed of pdit next time
   
    polysome1.groupTransForms(worker_n = 5) #parallel 
                                         
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
    fillPoly['class'] = np.array([-2])
    fillPoly['riboinfo'] = 0
    fillPoly['addNum'] = 2
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly()
    
#    tom_plot_vectorField(polysome1.transList, vectVisP['showTomo'], 
#                                  vectVisP['showClass'], vectVisP['polyNr'], vectVisP['onlySelected'],
#                                  vectVisP['repVectLen'],vectVisP['repVect'],
#                                  np.array([0,0,1]), '', '')   
    #load the filled up particle.star and get the filled up ribosomes
    particlesData = tom_starread('./sim_dropFillUp.star')
    fillupRibos = particlesData[particlesData['rlnClassNumber'] == -1]
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

