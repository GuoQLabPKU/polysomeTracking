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
    polysome1.sel[0]['minNumTransform'] = 50
    polysome1.creatOutputFolder()  
    polysome1.transForm['pixS'] = 3.42 # in Ang
    polysome1.transForm['maxDist'] = 342 # in Ang
    polysome1.calcTransForms(worker_n = 2) #the number of CPUs to process the data(#cpu == #tomograms)   
    polysome1.groupTransForms(gpu_list = [0]) # if you have GPUs, can do: polysome1.groupTransForms(gpu_list = [1,2])
    transListSel, selFolds = polysome1.selectTransFormClasses()  
    polysome1.alignTransforms()   
    polysome1.find_connectedTransforms()   
    polysome1.analyseTransFromPopulation()    
    #polysome1.analyseConfClasses()  #developing    
    #polysome1.genOutputList(transListSel, selFolds) #summary every transformation class
    #polysome1.generateTrClassAverages()
    #polysome1.genTrClassForwardModels()   
    
    polysome1.visResult()    
    polysome1.visPoly(lenPoly = lenPolyVis)   
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly() #you can also try to add more than one hypo ribo at the end of each poly by pass  fillupRiboN = 2   
    polysome1.visPoly(lenPoly = lenPolyVis)

def generateDeletPoly():
    setup()
    simStar = tom_starread('./sim.star')
    drop_index = 35
    ribo_info = simStar.iloc[35,:]
    print('drop the ribosome with euler angle:%.3f,%.3f,%.3f, and position:%.3f,%.3f,%.3f'%(ribo_info['rlnAngleRot'],
                                                                                           ribo_info['rlnAngleTilt'],
                                                                                           ribo_info['rlnAnglePsi'],
                                                                                           ribo_info['rlnCoordinateX'],
                                                                                           ribo_info['rlnCoordinateY'],
                                                                                           ribo_info['rlnCoordinateZ']))
    simStar.drop(index = [drop_index],inplace = True)
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"]  = ["_%s"%i for i in simStar.columns]
    tom_starwrite('./sim_drop.star', simStar, header)
    

if __name__ == '__main__':
    input_star = './sim_drop.star'
    clustThr = 5
    relinkWithoutSmallClasses = 0
    lenPolyVis = 10
    fillPoly = { }
    fillPoly['class'] = np.array([-2]) #-2:fillup ribosomes for all transformation classes
    fillPoly['riboinfo'] = 1
    fillPoly['addNum'] = 1
      
    if os.path.exists('cluster-sim_drop/run0/allTransforms.star'):
        os.remove('cluster-sim_drop/run0/allTransforms.star')   
    if os.path.exists('cluster-sim_drop/run0/scores/tree.npy'):
        os.remove('cluster-sim_drop/run0/scores/tree.npy')
        
    generateDeletPoly()
    runPoly(input_star, clustThr, relinkWithoutSmallClasses, lenPolyVis, fillPoly)
    teardown()