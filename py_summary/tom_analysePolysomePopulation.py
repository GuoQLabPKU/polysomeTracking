import numpy as np
import pandas as pd
from py_io.tom_starwrite import tom_starwrite
from py_transform.tom_angular_distance import tom_angular_distance
from py_transform.tom_average_rotations import tom_average_rotations


def analysePopulation(pairList): #the input shoule be a ptr
    
    classNr = pairList['pairClass'].values[0]
    
    stat = { }
    stat['classNr'] = classNr
    stat['num'] = pairList.shape[0] #number of the trans of one class
    
    vectStat = calcVectStat(pairList)
    angStat = calcAngStat(pairList)
    
    
    polyStat = calcPolyStat(pairList)
        
    stat['stdTransVect'] = vectStat['stdTransVect']
    stat['stdTransAng'] = angStat['stdTransAng']
    stat['numPolybg5'] = polyStat['numPolybg5']
    stat['numPolybg3'] = polyStat['numPolybg3']
    stat['numPolyMax'] = polyStat['numPolyMax']
    stat['numBranch'] = polyStat['numBranch']
    stat['tomoNrPolyMax'] = polyStat['tomoNrPolyMax']
    stat['polyIDMax'] = polyStat['polyIDMax']
    stat['meanTransVectX'] =vectStat['meanTransVectX']
    stat['meanTransVectY'] =vectStat['meanTransVectY']
    stat['meanTransVectZ'] =vectStat['meanTransVectZ']
    stat['meanTransAngPhi'] =angStat['meanTransAngPhi']
    stat['meanTransAngPsi'] =angStat['meanTransAngPsi']
    stat['meanTransAngTheta'] =angStat['meanTransAngTheta']
    
    return stat
    
        

def calcAngStat(pairList):
    angs = np.array([pairList['pairTransAngleZXZPhi'].values,
                     pairList['pairTransAngleZXZPsi'].values,
                     pairList['pairTransAngleZXZTheta'].values])
    angs = angs.transpose()
    meanAng,_,_ = tom_average_rotations(angs)
    lendiffAng = np.zeros(angs.shape[0])
    for i in range(angs.shape[0]):
        lendiffAng[i] = tom_angular_distance(angs[i,:], meanAng)
    stdTransAng = np.std(lendiffAng)
    stat = { }
    stat['meanTransAngPhi'] = meanAng[0]
    stat['meanTransAngPsi'] = meanAng[1]
    stat['meanTransAngTheta'] = meanAng[2]
    stat['stdTransAng'] = stdTransAng
    
    return stat
   
def calcVectStat(pairList):
    vects = np.array([pairList['pairTransVectX'].values,
                     pairList['pairTransVectY'].values,
                     pairList['pairTransVectZ'].values])
    vects = vects.transpose()
    if vects.shape[0] > 1:
        meanV = np.mean(vects, axis = 0 )
    else:
        meanV = vects    
    diffV = vects - meanV
    
    lendiffV = np.zeros(diffV.shape[0])
    for i in range(diffV.shape[0]):
        lendiffV[i] = np.linalg.norm(diffV[i,:])
    stdTransVect = np.std(lendiffV)
    stat = { }
    stat['meanTransVectX'] = meanV[0]
    stat['meanTransVectY'] = meanV[1]
    stat['meanTransVectZ'] = meanV[2]
    stat['stdTransVect'] = stdTransVect
    
    return stat

def calcPolyStat(pairList):
    allTomoID = pairList['pairTomoID'].values
    allLabel = pairList['pairLabel'].values
    allLabel_fix = np.fix(allLabel)
    allLabelU = np.unique(allLabel_fix)
    
    stat = { }
    if allLabelU[0] == -1.0:  #this is class == 0 OR no polysome tracking performed
        stat['numPolybg5'] = -1
        stat['numPolybg3'] = -1
        stat['numPolyMax'] = -1
        stat['tomoNrPolyMax'] = -1
        stat['polyIDMax'] = -1
        stat['numBranch'] = -1
    else:
        allNum = np.zeros(len(allLabelU), dtype = np.int) #1D array with number of polysomes in this class
        allPolyHasBranch = np.zeros(len(allLabelU), dtype = np.int)
        
        for i in range(len(allLabelU)):
            idx = np.where(allLabel_fix == allLabelU[i])[0]
            allNum[i] = len(idx) #the length of for each polysome with branch
            allPolyHasBranch[i] = np.sum(  (allLabel[idx] - allLabel_fix[idx]    ) > 0.05   ) > 0 #if this polysome has branch
        
        stat['numPolybg5'] = np.sum(allNum > 5)
        stat['numPolybg3'] = np.sum(allNum > 3)
        mPos = allNum.argmax() #not so accurate, because may two polysomes has the same length but only the first one kept in one tomo
        mVal = np.max(allNum)
        stat['numPolyMax'] = mVal #the longest polysomes in this class 
        stat['numBranch'] = np.sum(allPolyHasBranch)
        labelMax = allLabelU[mPos] #which polysome has the longest length
        tomoMax =  np.unique(allTomoID[np.where(allLabel == labelMax)[0]])
        if len(tomoMax) == 0:
            tomoMax = np.unique(allTomoID[np.where(allLabel_fix == labelMax)[0]])
        stat['tomoNrPolyMax'] = tomoMax[0] #the tomo with the longest polysome
        stat['polyIDMax'] = labelMax #the id of  the polysome (polyid_offset)
        
    return stat
        
def analysePopulationPerPoly(pairList):
    classNr = pairList['pairClass'].values[0]
    
    allTomoID = pairList['pairTomoID'].values
    allLabel = pairList['pairLabel'].values
    allLabel_fix = np.fix(allLabel)
    allLabelU = np.unique(allLabel_fix)
    
    stat = []
    if (allLabelU[0] == -1) | (classNr == 0): #don't analyse class 0 
        stat.append({})
        stat[0]['num'] = -1
        stat[0]['tomoNr'] = -1
        stat[0]['classNr'] = -1
        stat[0]['polyID'] = -1
        stat[0]['hasBranch'] = -1
        stat[0]['confClassVect'] = '-1'
        stat[0]['posInListVect'] = '-1'
        
        return stat
    else:
        for i in range(len(allLabelU)):
            idx = np.where(allLabel == allLabelU[i])[0] ##this is no branch(branch1)
            tmpLabel = allLabelU[i] + 0.1
            idxbB1 = np.where(allLabel == tmpLabel)[0]  ##this is with branch(branch2)
            if (len(idxbB1) > len(idx)):
                idx = idxbB1
            stat.append({})
            stat[i]['num'] = len(idx)
            if stat[i]['num'] == 20:
                print(' ')
            stat[i]['tomoNr'] = allTomoID[idx[0]]
            stat[i]['classNr'] = classNr
            stat[i]['polyID'] = np.round(pairList['pairLabel'].values[idx[0]])
            idx2 = np.where(allLabel_fix == allLabelU[i])[0]
            stat[i]['hasBranch'] = np.int(np.sum(  (allLabel[idx2] - allLabel_fix[idx2]    ) > 0.05   ) > 0)
            
            posInPolyVect = np.array([], dtype = np.int)
            confClassVect = np.array([], dtype = np.int)
            posInListVect = np.array([], dtype = np.int)
            
            for ii in range(len(idx)):#only analysis one branch which longer than another branch
                posInPolyVect = np.concatenate((posInPolyVect, np.array([pairList['pairPosInPoly1'].values[idx[ii]],
                                                                        pairList['pairPosInPoly2'].values[idx[ii]] ]) ))
                confClassVect = np.concatenate((confClassVect, np.array([pairList['pairClass1'].values[idx[ii]],
                                                                         pairList['pairClass2'].values[idx[ii]]], dtype = np.int)))
                posInListVect = np.concatenate((posInListVect, np.array([pairList['pairIDX1'].values[idx[ii]],
                                                                         pairList['pairIDX2'].values[idx[ii]] ])))
            _, indices = np.unique(posInPolyVect, return_index = True)
            if len(indices) < stat[i]['num']:
                stat[i]['num'] = len(indices)  #the unique polysome w/o any branch
            stat[i]['confClassVect'] = '-'.join([str(i) for i in confClassVect[indices]])                      
            stat[i]['posInListVect'] =  '-'.join([str(i) for i in posInListVect[indices]])
            
    return stat
                
                                                                        
                                                                                 
def sortStatPoly(statePerPolyTmp):
    #merge the polysome information and then sort the polysomes info by the 'num'
    tmpList = [ ]
    for single_class in statePerPolyTmp:
        tmpList.extend(single_class)
    tmp_frame = { }
    keys_list = tmpList[0].keys()
    for k in keys_list:
        tmp_frame[k] = [C[k] for C in tmpList]
        
    tmp_frame = pd.DataFrame(tmp_frame)
    tmp_frame = tmp_frame.sort_values(['num'], ascending = False)  
    return tmp_frame

def sortStat(stat):
    polyFact = [((single_class['numPolybg3'] + single_class['numPolybg5'])*10000 + single_class['num']) \
    for single_class in stat]
    idx = np.argsort(polyFact)[::-1]
    stat = [stat[i] for i in idx]
    #make the form for dataframe
    stat_frame = { }
    keys_list = stat[0].keys()
    for k in keys_list:
        stat_frame[k] = [C[k] for C in stat]
        
    stat_frame = pd.DataFrame(stat_frame)
        

    return stat_frame
    
    
def writeOutputStar(stat, statPoly, outputFolder = ''): #the two inputs shoule be dataframe data structure
    if outputFolder != '':
        header = { }
        header["is_loop"] = 1
        header["title"] = "data_"
        header["fieldNames"]  = ["_%s"%i for i in stat.columns]
        tom_starwrite('%s/statPerClass.star'%outputFolder, stat, header)
        
        del header
        header = { }
        header["is_loop"] = 1
        header["title"] = "data_"
        header["fieldNames"]  = ["_%s"%i for i in statPoly.columns]  
        tom_starwrite('%s/statPerPoly.star'%outputFolder, statPoly, header)
        
def genOutput(stat, minTransMembers):
    if stat.shape[0] > 20: #the number of this represent the class numbers!
        stat = stat[stat['num'] > minTransMembers] #each row represent one class, this number represent the #transformation in this class
        stat.reset_index(inplace = True, drop = True)

    select_col = ['classNr',  'num', 'stdTransVect', 'stdTransAng', 
                'numPolybg5', 'numPolybg3', 'numPolyMax', 'numBranch']
    #print the whole data set
    for i in select_col:
        print(i, end = "\t")
    print('\t')
    for row in range(stat.shape[0]):
        print("%d\t%d\t%.1f\t%.1f\t%d\t%d\t%d\t%d"%(stat['classNr'].values[row], stat['num'].values[row], stat['stdTransVect'].values[row],
                                                        stat['stdTransAng'].values[row], stat['numPolybg5'].values[row],
                                                        stat['numPolybg3'].values[row], stat['numPolyMax'].values[row],
                                                        stat['numBranch'].values[row]))
        
        
    if stat.shape[0] > 20:
        print('only classes with more than %d transforms showed!'%minTransMembers)
        
    
    

        
    
     
    
    
    
                                                                   
                                                                                 
                                                                                 
                                                                                 
                
      
        
        
            
        
        
    
    
    