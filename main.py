import numpy as np
from py_test.addRmPoly import setup, teardown
from polysome_class.polysome import Polysome
from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite


###run just for test?###
using_simulation = 0 #1: just run simulation data/0:run user provided data
noizeDegree = 0 #the factor to add noise. 0:no noise 
branch = 1 #1:generate polysomes with branches; 0:generate polysomes w/o branches

#####BASIC PARAMTERS SETTING########
input_star =  './dataStar/Julia_share/Cm20181109_warp107_relionjob284_run_it035_data.star'
pixelSize = 1.7 # in Ang, the pixelsize of particle.star
cluster_threshold = 35
relinkWithoutSmallClasses = 1 #clean the classes with #transform<minNumTransformPairs. 0:switch off cleaning
minNumTransformPairs = 100  #if relinkWithoutSmallClasses == 0, this parameter is useless
cpuN = 5 #number of cores for parallel computation. 1:switch off parallel computation
gpuList = [0]  #leave None if no gpu available. Else input gpuID list like [0]/[0,1]
if_vispoly = 1 #1:switch on. This will plot each polysomes which length>10. It is good 
               #for vislization, but need manually close the figures.
               #If you give a positive value>1. Then the polysomes with length>if_vispoly
               #will be displayed
#####ADVANCED PARAMETERS SETTING######
maxDist = pixelSize*200 # in Ang
fillUpPoly_class = np.array([-1]) #which cluster class to fill up ribosomes. -1:all classes
fillUpPoly_addNum = 1 #number of ribosomes added in each tail of polysome 0:switch off filling up
fillUpPoly_riboinfo = 1 #if print out the information of filled up ribosomes. 0:switch off


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
    

if __name__ == '__main__':   
    if using_simulation:
        input_star = './sim_drop.star'
        pixelSize = 3.24
        cluster_threshold = 5
        minNumTransformPairs = 5
        if_vispoly = 0
        generateDeletPoly(noizeDegree, branch)
    
    fillUpPoly = { }
    fillUpPoly['class'] = fillUpPoly_class 
    fillUpPoly['riboinfo'] = fillUpPoly_riboinfo 
    fillUpPoly['addNum'] = fillUpPoly_addNum 
    runPoly(input_star, pixelSize, maxDist, cluster_threshold, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillUpPoly, cpuN, gpuList, if_vispoly) 
    
    if using_simulation:
            teardown()