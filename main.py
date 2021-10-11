import numpy as np
from polysome_class.polysome import Polysome

#####BASIC PARAMTERS SETTING########
input_star =  './dataStar/toxo_4tomos_refine3D_15A.star'
run_time = 'run0'
pixelSize = 3.38 # in Ang, the pixelsize of particle.star
cluster_threshold = 35
relinkWithoutSmallClasses = 1 #clean the classes with #transform<minNumTransformPairs. 0:switch off cleaning
minNumTransformPairs = 50  #if relinkWithoutSmallClasses == 0, this parameter is useless
remove_branches=0 #1:branch removing; 0:switch off

####PARALLEL PARAMETERS####
cpuN = 10 #number of cores for parallel computation. 1:switch off parallel computation
gpuList = [0]  #leave None if no gpu available. Else input gpuID list like [0]/[0,1]

#####VISUAL PARAMETERS######
vectorfield_plotting = 'basic' #advance:show detail information of polysomes
show_longestPoly = 1 #if plot and save longest polysomes for specific transform class. 0:switch off
longestPoly_ClassNr = np.array([-1]) #plot the longest polysome of specific class. negative value: plot the whole classes
                                     #for examle: np.array([1,2]) => class1 & class2
                                     
if_vispoly = 0 #1:switch on. This will plot each polysomes which length>10. It is good 
               #for vislization, but need manually close the figures.
               #If you give a positive value>1. Then the polysomes with length>if_vispoly
               #will be displayed   
               
#####ADVANCED PARAMETERS SETTING######
maxDist = 3.38*100 # in Ang
fillUpPoly_classNr = np.array([-1]) #which cluster class to fill up ribosomes. -1:all classes
fillUpPoly_addNum = 0 #number of ribosomes added in each tail of polysome  0:switch off filling up step
fillUpPoly_riboInfo = 1 #if print out the information of filled up ribosomes. 0:switch off
fillUpPoly_model = 'lognorm' #fitting distribution(genFit:based on data/lognorm:base on lognorm model)/max:no model fitting(only for small dataset)
fillUpPoly_threshold = 0.05 #threshold to accept filled up ribosomes

def runPoly(input_star, run_time, pixelSize, maxDist, clustThr, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillPoly, cpuN, gpuList, remove_branches, vetorPlot_type, show_longestPoly,
            show_PolyClassNr, if_vispoly):

    polysome1 = Polysome(input_star = input_star, run_time = run_time)
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
    polysome1.link_ShortPoly(remove_branches, cpuN) #default, this step will add ribosomes and then do polysome tracking
    polysome1.analyseTransFromPopulation('','',1)
    
    #polysome1.noiseEstimate()
    
    polysome1.vis['vectField']['type'] = vetorPlot_type
    polysome1.vis['longestPoly']['render'] = show_longestPoly
    polysome1.vis['longestPoly']['showClassNr'] = show_PolyClassNr
    polysome1.visResult()   
    polysome1.visLongestPoly()
    
    if if_vispoly > 0:
        polysome1.vis['vectField']['type'] = 'advance'
        polysome1.visPoly(if_vispoly)
    
if __name__ == '__main__':   
    
    fillUpPoly = { }
    fillUpPoly['classNr'] = fillUpPoly_classNr 
    fillUpPoly['riboInfo'] = fillUpPoly_riboInfo 
    fillUpPoly['addNum'] = fillUpPoly_addNum 
    fillUpPoly['fitModel'] = fillUpPoly_model 
    fillUpPoly['threshold'] = fillUpPoly_threshold

    runPoly(input_star, run_time, pixelSize, maxDist, cluster_threshold, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillUpPoly, cpuN, gpuList, remove_branches, vectorfield_plotting, 
            show_longestPoly, longestPoly_ClassNr, if_vispoly) 
    
