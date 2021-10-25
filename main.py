import numpy as np
from polysome_class.polysome import Polysome

#####BASIC PARAMTERS SETTING########
input_star =  './dataStar/ecoli/sel5TomosEcoli.star'
run_time = 'run6_threshold20_relink01percent' #the subfolder name 
pixelSize = 3.52 #in Ang, the pixelsize of input starfile
cluster_threshold = 20
relinkWithoutSmallClasses = 1 #clean the class with #transforms<minNumTransformPairs. 0:switch off cleaning
minNumTransformPairs = -1  #select classes with #transforms>minNumTransformPairs for storage and relink
remove_branches=0 #1:branch removing; 0:switch off
average_particles = 1 #average particles from each class #1.switch on
####PARALLEL COMPUTATION PARAMETERS####
cpuN = 15 #number of cores for parallel computation. 1:switch off parallel computation
gpuList = None  #leave None if no gpu available. Else input gpuID list like [0]//[0,1]
avg_cpuNr = 35 #the number of CPU when average particles by relion if average_particles==1
#####VISUAL PARAMETERS######
vectorfield_plotting = 'basic' #advance:show detail information of polysomes
show_longestPoly = 1 #plot and save the longest polysome. 0:switch off
longestPoly_ClassNr = np.array([-1]) #plot the longest polysome for specific class. negative value: plot for all classes
                                     #for examle: np.array([1,2]) => plot for class1 & class2
                                     
if_vispoly = 0 #1:switch on. Plot polysomes with length>10. Only for vislization.
               #If you give a positive value>1. Then the polysomes with length>if_vispoly
               #will be displayed   
               
#####ADVANCED PARAMETERS SETTING######
maxDist = 3.52*50 # in Ang.  #the searching region of adjacent ribosomes
link_depth = 2 #the depth for linking adjacent transforms into longer polysomes. 0:clean branches
fillUpPoly_classNr = np.array([-1]) #which cluster class for filling up ribosomes. -1:all classes
fillUpPoly_addNum = 0 #number of ribosomes added in each tail of polysome  0:switch off filling up step
fillUpPoly_riboInfo = 1 #if print out the information of filled up ribosomes. 0:switch off
fillUpPoly_model = 'lognorm' #fit distribution(genFit:based on data/lognorm:base on lognorm model)/max:no model fitting
fillUpPoly_threshold = 0.05 #threshold to accept filled up ribosomes

avg_pixS = 3.42  #the pixel size of particles for relion alignment
avg_minPart = 50 #the minmual number particles requirement for average
avg_maxRes = 20 #the parameter of relion
avg_callByPython = 0 #if use python to call relion.0: generate one script for relion, 1:python call relion

def runPoly(input_star, run_time, pixelSize, maxDist, link_depth, clustThr, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillPoly, cpuN, gpuList, remove_branches, vetorPlot_type, show_longestPoly,
            show_PolyClassNr, if_vispoly, if_avg, avg):

    polysome1 = Polysome(input_star = input_star, run_time = run_time)
    polysome1.transForm['pixS'] = pixelSize 
    polysome1.transForm['maxDist'] = maxDist 
    polysome1.transForm['branchDepth'] = link_depth
    polysome1.classify['clustThr'] = clustThr
    polysome1.classify['relinkWithoutSmallClasses'] = relinkWithoutSmallClasses
    polysome1.sel[0]['minNumTransform'] = minNumTransformPairs
    
    
    polysome1.creatOutputFolder()  
    polysome1.calcTransForms(worker_n = cpuN)
    polysome1.groupTransForms(worker_n = cpuN, gpu_list = gpuList) 
    transListSel, selFolds = polysome1.selectTransFormClasses() 
    polysome1.genOutputList(transListSel, selFolds)
    polysome1.alignTransforms()   
    polysome1.analyseTransFromPopulation('','',0)           
    polysome1.fillPoly = fillPoly
    polysome1.link_ShortPoly(remove_branches, cpuN) 
    polysome1.analyseTransFromPopulation('','',1)    
    polysome1.noiseEstimate()
    
    polysome1.vis['vectField']['type'] = vetorPlot_type
    polysome1.vis['longestPoly']['render'] = show_longestPoly
    polysome1.vis['longestPoly']['showClassNr'] = show_PolyClassNr
    polysome1.visResult()   
    polysome1.visLongestPoly()
    
    if if_vispoly > 0:
        polysome1.vis['vectField']['type'] = 'advance'
        polysome1.visPoly(if_vispoly)
    
    #average particles subset using relion
    if if_avg:
        polysome1.avg = avg
        polysome1.generateTrClassAverages()
    
if __name__ == '__main__':   
    
    fillUpPoly = { }
    fillUpPoly['classNr'] = fillUpPoly_classNr 
    fillUpPoly['riboInfo'] = fillUpPoly_riboInfo 
    fillUpPoly['addNum'] = fillUpPoly_addNum 
    fillUpPoly['fitModel'] = fillUpPoly_model 
    fillUpPoly['threshold'] = fillUpPoly_threshold

    avg = { }
    avg['filt'] = { }
    avg['filt']['minNumPart'] = avg_minPart 
    avg['filt']['maxNumPart'] = np.inf
    avg['pixS'] = avg_pixS
    avg['maxRes'] = avg_maxRes
    avg['cpuNr'] = avg_cpuNr
    avg['callByPython'] = avg_callByPython
   
    runPoly(input_star, run_time, pixelSize, maxDist, link_depth, cluster_threshold, relinkWithoutSmallClasses, 
            minNumTransformPairs, fillUpPoly, cpuN, gpuList, remove_branches, vectorfield_plotting, 
            show_longestPoly, longestPoly_ClassNr, if_vispoly,average_particles, avg) 
    
