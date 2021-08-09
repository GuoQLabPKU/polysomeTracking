import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from py_io import tom_starread
from py_io import tom_starwrite
def tom_findPattern(vbos, sigLevel = 97.5, nrSimulations = 1000, outputFolder = '', filterSt = ''):
    '''
    
    TOM_FINDPATTERN finds repetative patterns in a vector

    resSt=tom_findPattern(vobs,sigLevel,nrSimulations,outputFolder,filterSt)

    PARAMETERS

    INPUT
        vobs                            vector of observations
        sigLevel                        level of significant
        nrSimulations                   number of simulations   
        outputFolder                    ('') folder for ouput  
        filterSt                        ('') structure to filter polysome file

  
    OUTPUT
        resSt                            result structure 

    EXAMPLE

        tom_findPattern({[1 2 1 2 1 2 1 2 3]})
 
         filterSt.filterStTmp=5;
         filterSt.num.value=3;
         filterSt.num.operator='>';
         tom_findPattern('test.star',98.5,500,'',filterSt);

    '''
    patternFlag = 'permutation'
    print('Pattern analysis started')
    
    filterStTmp = { }
    if len(filterSt) == 0:
        filterStTmp[0] = -1
    else:
        for i in range(len(filterSt['classNr'])):
            filterStTmp[i] = filterSt
            filterStTmp[i]['classNr'] = filterSt['classNr'][i]
            #therefor, the filterStTmp contain single class from filter
    
    for i in len(filterStTmp): #each one class
        if  'classNr' in filterStTmp[0].keys():
            if len(filterSt['classNr'] > 1):
                outputFolderTmp = "%s/cl%d/"%(outputFolder, filterSt['classNr'][i])
        else:
            outputFolderTmp = outputFolder
        findPatternOneClass(vbos, sigLevel, nrSimulations, outputFolderTmp, filterStTmp[i], patternFlag)

def findPatternOneClass(vbos, sigLevel, nrSimulations, outputFolder, filterSt, patternFlag):
    if type(vbos).__name__ == 'str':
        fName = vbos
        vbos = getObsVectFromFile(vbos, filterSt)
    
    pattern = genPattern(vbos, patternFlag)
    resSt = searchPattern(vbos, pattern, ' in observation vector')
    vbosRandM  = genRandomMatrix(vbos, nrSimulations)
    resStRand = searchPattern(vbosRandM, pattern, 'in simulation')
    statM = calcStatFromRandM(resStRand, sigLevel)
    plotResults(statM, resSt, outputFolder)
    genOutputTable(statM, resSt, outputFolder)

def genOutputTable(statM, resSt, outputFolder):
    if len(outputFolder) > 0:
        header = { }
        header["is_loop"] = 1
        header["title"] = "data_"
        header["fieldNames"]  = ['_counts', '_sigcounts', '_isSignificant', '_pattern']
        
        idbgZ = np.where(resSt['fCount'] > 0)[0]
        sigCount = statM['sig'][idbgZ]
        isSig = resSt['fCount'][idbgZ] > statM['sig'][idbgZ].transpose()
        counts = resSt['fCount'][idbgZ]
        pattern = statM['Xlabel'][idbgZ]
        
        tmp = isSig*1000000 + counts
        idxS = np.argsort(tmp)[::-1]
        
        stat = pd.DataFrame({})
        if  not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        stat['counts'] = counts[idxS]
        stat['sigcounts'] = sigCount[idxS]
        stat['isSignificant'] = np.double(isSig[idxS])
        stat['pattern'] = pattern[idxS]
        tom_starwrite('%s/patternCounts.star'%outputFolder, stat, header)
        

def plotResults(statM, resSt, outputFolder):
    plt.figure()
    plt.title('clustering')
    idx = np.where(resSt['fCount'] > 0)[0]
    plt.plot(statM['patternNr'][0:len(idx)],   statM['mean'][idx] , 'r+')
    plt.plot(statM['patternNr'][0:len(idx)], statM['mean'][idx], 'r-'  )
    
    plt.plot(statM['patternNr'][0:len(idx)], statM['sig'][idx], 'r--')
    plt.plot(statM['patternNr'][0:len(idx)], statM['sig'][idx], 'r++')

    for i in range(len(statM['patternNr'][0:len(idx)])):
        idxTmp = np.arange(0,len(idx))
        pNr = statM['patternNr'][idxTmp[i]]
        x = [pNr, pNr]
        y = [statM['mean'][idx[i]],   statM['sig'][idx[i]]]
        plt.plot(x,y,'r-')
    plt.plot(statM['patternNr'][0:len(idx)], resSt['fCount'][idx].reshape(1,-1)[0], 'bo')
    plt.plot(statM['patternNr'][0:len(idx)], resSt['fCount'][idx].reshape(1,-1)[0], 'b-')
    
    plt.xticks(statM['patternNr'][0:len(idx)],
               statM['Xlabel'][idx])
    
    if len(outputFolder) > 0:
        if  not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        plt.savefig('%s/patternCounts.fig'%outputFolder, dpi = 300)
    plt.show()
    plt.close()
    

def calcStatFromRandM(resStRand, sig):
    nStd = 3
    stat = { }
    stat['patternNr'] = np.arange(resStRand[0]['fCount'].shape[0])
    fCount_array = resStRand[0]['fCount']
    for single_resSt in resStRand[1:]:
        fCount_array = np.concatenate((fCount_array, single_resSt['fCount']) ,axis = 1)
    stat['mean'] = np.mean(fCount_array, axis = 1)  #1D array
    stat['std'] = np.std(fCount_array, axis = 1, ddof = 1)
    stat['errf'] = stat['mean'] + nStd*stat['std']
    stat['sig'] = calcSignificantCounts(fCount_array, sig)  #sig is one 1D array
    
    tmpPattern = resStRand[0]['pattern']
    stat['Xlabel'] = [ ]
    for i in range(len(tmpPattern)):
        stat['Xlabel'].append('-'.join([str(i) for i in tmpPattern[i]]))
    return stat
                
def calcSignificantCounts(countsPatternVsRepeats, sigLevel):
    numRepeat = countsPatternVsRepeats.shape[1]
    numPattern = countsPatternVsRepeats.shape[0]
    uCountsAllPattern = np.unique(countsPatternVsRepeats)     
    sigLevelOccurence = numRepeat*(1-(sigLevel/100)) 
    
    allsig = np.zeros(numPattern, dtype = np.int)
    for patNr in range(numPattern):
        countsOnePattern = countsPatternVsRepeats[patNr,:]
        occurenceDistributionOnPattern = calcCountDist(countsOnePattern, uCountsAllPattern)  #2D with nx1 array
        occurCum = 0
        for i in np.arange(len(occurenceDistributionOnPattern), -1, -1):
            occurCum += occurenceDistributionOnPattern[i,0]
            if occurCum > sigLevelOccurence:
                break
        if i < len(uCountsAllPattern):
            allsig[patNr] = np.mean([uCountsAllPattern[i], uCountsAllPattern[i+1]])
        if i == len(uCountsAllPattern):
            allsig[patNr] = uCountsAllPattern[i]
    return allsig #1D array
            

def calcCountDist(countsOnePattern, uCountAllP):
    countDistribution = np.zeros(len(uCountAllP),1)
    for i in range(len(uCountAllP)):
        countDistribution[i,0] =    len(np.where(countsOnePattern == uCountAllP[i])[0])
    return countDistribution
    
    
def genRandomMatrix(vbos, numRepeats):
    allCl = np.array([],dtype = np.int)
    for i in range(len(vbos)):
        allCl = np.concatenate((allCl, vbos[i]))
    allClU = np.unique(allCl)
    
    obsPool = np.array([ ], dtype = np.int)
    oStart = 0
    obsLenVect = [ ]
    obsLenStartStop = [ ]
    for i in range(len(vbos)):
        obsLenVect.append(len(vbos[i]))  #one element， one polysome
        oStop = oStart + len(vbos[i]) - 1
        obsLenStartStop.append((oStart, oStop))
        obsPool = np.concatenate((obsPool, vbos[i]))
        oStart = oStop + 1
    
    vbosRand = [ ]
    obsRandM = [ ]
    for i in range(numRepeats):
        randTmpIdx = np.random.permutation(len(obsPool))
        for ii in range(len(obsLenVect)):
            randTmp = obsPool[randTmpIdx[obsLenStartStop[ii][0]:obsLenStartStop[ii][1]  +1              ]]
            vbosRand.append(randTmp)
        obsRandM.append(vbosRand)
    return obsRandM
            
        
            
        
def searchPattern(vbos, pattern, tag):
    print('searching pattern %s'%tag)
    resSt = [ ]
    for i in range(len(vbos)):
        resSt.append(searchOneObs(vbos[i], pattern))
    
    return resSt  #resSt is one list with dict stored , each dict is one polysome from the same class transform

def searchOneObs(vbos, pattern):
    fCount = np.zeros((len(pattern), 1))
    for ipat in range(len(pattern)):
        pat = "-".join([str(i) for i in pattern[ipat]])
        for ibos in range(len(vbos)):
            obs = "-".join([str(i) for i in vbos[ibos]])
            nrFound = len([i.span()[0] for i in re.finditer(pat, obs)])
            fCount[ipat, 0] = fCount[ipat, 0] + nrFound
    resSt = { }
    resSt['fCount'] = fCount
    resSt['pattern'] = pattern
    return resSt
            
        
def getObsVectFromFile(fName, filterSt):
    if '.star' in fName:
        st = tom_starread(fName)
        
        vUsed = np.ones((1, st.shape[0]  ))
        if 'classNr' in filterSt.keys():
            print('analysing pattern for class:%d'%filterSt['classNr'])
            classNr = st['classNr'].values
            vUsed = vUsed*(classNr == filterSt['classNr'])  #only keep the filterSt.classNr == classNr we want to analyse
        if 'filt'  in filterSt.keys():
            num = st['num'].values
            expr = 'num %s %d'%(filterSt['filt']['operator'], filterSt['filt']['value']) #only analysis len(polysome) > 3
            vtmp = eval(expr)
            vUsed = vUsed*(vtmp > 0)
            
        idx = np.where(vUsed == 1)[0] #class and also the len(polysome)
        vbos = [ ]
        for i in idx:
            vbos.append([int(i) for i in st['confClassVect'].values[i].split("-")])
        return vbos #return the class of each ribosomes 
    else:
        print('Error: unsuported input data type!')
        os._exit(-1)
           
def genPattern(vbos, patternFlag):
    allCl = np.array([],dtype = np.int)
    lenVobs = [ ]
    for i in range(len(vbos)):
        allCl = np.concatenate((allCl, vbos[i]))
        lenVobs.append(len(vbos[i]))
    allClU = np.unique(allCl)
    nrCmb = 0
    for i in range(2, len(allClU)+1):
        nrCmb += np.max(allClU)**i  ###WHY?
    print('Found %d ribosomes classes ===> %d patterns will be generated!'%(len(allClU), len(nrCmb )))
    zz = 1
    pattern = [ ]  #three cycles,very time consuming
    for i in range(2,len(allClU)+1):
        nrCmb = np.max(allClU)**i
        for ii in range(1,nrCmb+1):
            if np.max(allClU) > 36:
                print('Warning: the class number is too large to convert base!')
            patternTmp = np.base_repr(ii - 1, max(allClU), i)
            patV = [int(i)+1 for i in patternTmp ]
            pattern.append(patV)
            del patV
            zz += 1
    return pattern
        