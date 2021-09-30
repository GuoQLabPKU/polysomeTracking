import numpy as np
import sys
sys.path.append('/lustre/Data/jiangwh/polysome/python_version/polysome/')
from polysome_class.polysome import Polysome
from py_test.addRmPoly import setup, teardown
from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite

###PARAMETERS FOR SIMULATED POLYSOMES GENERATION###
noizeDegree = 0 #the factor to add noise in the major polysome branch. 0:no noise 
branch = 1 #1:generate polysomes with branches; 0:polysomes w/o branches

def generateDeletPoly(noizeDegree, branch):
    setup(None, noizeDegree, branch)
    simStar = tom_starread('./sim.star')
    simStar_ = simStar['data_particles']
    drop_index = [30]
    ribo_info = simStar_.iloc[drop_index,:]
    for i in range(ribo_info.shape[0]):
        print('drop the ribosome with euler angle:%.3f,%.3f,%.3f, and position:%.3f,%.3f,%.3f'%(ribo_info['rlnAngleRot'].values[i],
                                                                                                ribo_info['rlnAngleTilt'].values[i],
                                                                                                ribo_info['rlnAnglePsi'].values[i],
                                                                                                ribo_info['rlnCoordinateX'].values[i],
                                                                                                ribo_info['rlnCoordinateY'].values[i],                                                                                                
                                                                                                ribo_info['rlnCoordinateZ'].values[i]))
    print('')
    simStar_.drop(index = drop_index, inplace = True)
    simStar['data_particles'] = simStar_
    tom_starwrite('./sim_drop.star', simStar)
    
    
def runPoly(input_star, pixelSize, maxDist, clustThr, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillPoly, cpuN, gpuList, if_vispoly):

    polysome1 = Polysome(input_star = input_star, run_time = 'run0')
    polysome1.transForm['pixS'] = pixelSize 
    polysome1.transForm['maxDist'] = maxDist 
    polysome1.classify['clustThr'] = clustThr
    polysome1.classify['relinkWithoutSmallClasses'] = relinkWithoutSmallClasses
    polysome1.sel[0]['minNumTransform'] = minNumTransformPairs
    polysome1.creatOutputFolder()  
    polysome1.calcTransForms(worker_n = cpuN) #the number of CPUs to process the data(#cpu == #tomograms)   
    polysome1.groupTransForms(worker_n = cpuN, gpu_list = gpuList) # if you have GPUs, can do: polysome1.groupTransForms(gpu_list = [1,2])
    transListSel, selFolds = polysome1.selectTransFormClasses() #clean transformation
    polysome1.alignTransforms()   
    polysome1.analyseTransFromPopulation('','',0)           
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly() #default, this step will add ribosomes and then do polysome tracking
    polysome1.analyseTransFromPopulation('','',1)
    polysome1.visResult()    
    if if_vispoly > 0:
        polysome1.visPoly(if_vispoly)
        
if __name__ == '__main__':   

    input_star = './sim_drop.star'
    pixelSize = 3.24
    cluster_threshold = 5
    minNumTransformPairs = 5
    if_vispoly = 10
    
    fillUpPoly = { }
    fillUpPoly['class'] = np.array([-1]) 
    fillUpPoly['riboinfo'] = 1 
    fillUpPoly['addNum'] = 1 
    
    generateDeletPoly(noizeDegree, branch)   
    runPoly(input_star, pixelSize, pixelSize*100, cluster_threshold, 1, 
            minNumTransformPairs, fillUpPoly, 1, None, if_vispoly)    
    teardown()