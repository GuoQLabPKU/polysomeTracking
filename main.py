import os 
import numpy as np
from py_test.addRmPoly import setup, teardown
from polysome_class.polysome import Polysome
from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite


def runPoly(input_star, clustThr, relinkWithoutSmallClasses, lenPolyVis, fillPoly):
    polysome1 = Polysome(input_star = input_star, run_time = 'run0')
    polysome1.classify['clustThr'] = clustThr
    polysome1.classify['relinkWithoutSmallClasses'] = relinkWithoutSmallClasses
    polysome1.sel[0]['minNumTransform'] = 5
    polysome1.creatOutputFolder()  
    polysome1.transForm['pixS'] = 3.24 # in Ang
    polysome1.transForm['maxDist'] = 3.24*100 # in Ang
    polysome1.calcTransForms(worker_n = 2) #the number of CPUs to process the data(#cpu == #tomograms)   
    polysome1.groupTransForms(worker_n = 5) # if you have GPUs, can do: polysome1.groupTransForms(gpu_list = [1,2])
    transListSel, selFolds = polysome1.selectTransFormClasses() #clean transformation
    polysome1.alignTransforms()   
    polysome1.analyseTransFromPopulation()        
    polysome1.visResult()  
    
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly() #default, this step will add ribosomes and then do polysome tracking
    polysome1.visResult()
    polysome1.analyseTransFromPopulation()  

def generateDeletPoly():
    setup(None,noizeDregree = 0, branch = 1) #branch = 0:generate polysomes w/o branch
    simStar = tom_starread('./sim.star')
    drop_index = [30,31]
    ribo_info = simStar.iloc[drop_index,:]
    for i in range(ribo_info.shape[0]):
        print('drop the ribosome with euler angle:%.3f,%.3f,%.3f, and position:%.3f,%.3f,%.3f'%(ribo_info['rlnAngleRot'].values[i],
                                                                                                ribo_info['rlnAngleTilt'].values[i],
                                                                                                ribo_info['rlnAnglePsi'].values[i],
                                                                                                ribo_info['rlnCoordinateX'].values[i],
                                                                                                ribo_info['rlnCoordinateY'].values[i],
                                                                                               ribo_info['rlnCoordinateZ'].values[i]))
    print('')
    simStar.drop(index = drop_index,inplace = True)
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"]  = ["_%s"%i for i in simStar.columns]
    tom_starwrite('./sim_drop.star', simStar, header)
    

if __name__ == '__main__':
    input_star = './sim_drop.star'
    clustThr = 5
    relinkWithoutSmallClasses = 1
    lenPolyVis = 5
    
    #parameters for fillup
    fillPoly = { }
    fillPoly['class'] = np.array([-1]) #-1:fillup all transformation classes
    fillPoly['riboinfo'] = 1 #if print the information of filled up ribosomes
    fillPoly['addNum'] = 2 #add one ribosomes at the end of each polysome
      
    if os.path.exists('cluster-sim_drop/run0/allTransforms.star'):
        os.remove('cluster-sim_drop/run0/allTransforms.star')   
    if os.path.exists('cluster-sim_drop/run0/scores/tree.npy'):
        os.remove('cluster-sim_drop/run0/scores/tree.npy')
        
    generateDeletPoly()
    runPoly(input_star, clustThr, relinkWithoutSmallClasses, lenPolyVis, fillPoly)
    teardown()