import numpy as np
import warnings
import pandas as pd


from py_io.tom_starread import tom_starread, generateStarInfos
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_extractData import tom_extractData
from py_link.tom_connectGraph import tom_connectGraph
from py_mergePoly.tom_extendPoly import checkRibo
from py_mergePoly.tom_addTailRibo import updateParticle
from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate

                    
def tom_interpolatRibos(allTransListFile, stateFile, particleDetect, particleOut, interpoltaN = 10, classList = None, 
                        polyLenRange = None):
    '''
    TOM_INTERPOLATRIBOS interpolate ribosomes in the head & tail diretion 
    of each polysome with length(polyLenRange) and (interpoltaN) particles 
    in the class (classList), and will be filtered if already detected 
    in the particleDetect file. Addition is aviliabel by rotate and shift 
    in the stateFile
    '''
    interplotRiboStruct = pd.DataFrame({})
    if isinstance(particleDetect, str):
        particleSt = tom_extractData(particleDetect)
        particleData_ = tom_starread(particleDetect)
        particleData = particleData_['data_particles']
    if isinstance(allTransListFile, str):
        allTransList_ = tom_starread(allTransListFile)
        allTransList = allTransList_['data_particles']
    if isinstance(stateFile, str):
        state_summary_ = tom_starread(stateFile)
        classSummary  = state_summary_['data_particles']              
    if classList is None:
        classList = np.unique(allTransList['pairClass'].values)
    if polyLenRange is None:
        polyLenRange = (0, np.inf)
    
    statePolyAll_forFillUp = tom_connectGraph(allTransList) 
    statePolyAll_forFillUp = statePolyAll_forFillUp[statePolyAll_forFillUp['polylen_riboNr']>=polyLenRange[0]]
    statePolyAll_forFillUp = statePolyAll_forFillUp[statePolyAll_forFillUp['polylen_riboNr']<=polyLenRange[1]]
    
    for singleC in classList:        
        if singleC == -1:
            warnings.warn('no transformation class deteced!')
            return
        if singleC == 0:
            continue    
        
        avgShift = classSummary[classSummary['classNr'] == singleC].loc[:,
                                   ['meanTransVectX','meanTransVectY','meanTransVectZ']].values[0] #1D array
        avgRot = classSummary[classSummary['classNr'] == singleC].loc[:,
                                   ['meanTransAngPhi','meanTransAngPsi','meanTransAngTheta']].values[0] #1D array
        
        statePolyAll_forFillUpSingleClass = statePolyAll_forFillUp[statePolyAll_forFillUp['pairClass'] == singleC]                 
        #get the head/tail ribo position & angle information
        tailRiboIdx = statePolyAll_forFillUpSingleClass[statePolyAll_forFillUpSingleClass['ifWoOut'] == 1]['pairIDX'].values
        headRiboIdx = statePolyAll_forFillUpSingleClass[statePolyAll_forFillUpSingleClass['ifWoIn'] ==  1]['pairIDX'].values
        print('detect %d begin ribosomes in the polysome'%len(headRiboIdx))
        print('detect %d end ribosomes in the polysome'%len(tailRiboIdx))
        #colllect the information of the tail& head ribosomes of each polysome
        tailRiboInfo  = np.zeros((len(tailRiboIdx), 8))
        headRiboInfo = np.zeros((len(headRiboIdx), 8)) 
        
        tailRiboInfo[:,0] = tailRiboIdx
        headRiboInfo[:,0] = headRiboIdx
        tailRiboInfo[:,1] = statePolyAll_forFillUpSingleClass[statePolyAll_forFillUpSingleClass['ifWoOut'] == 1]['pairTomoID'].values 
        headRiboInfo[:,1] = statePolyAll_forFillUpSingleClass[statePolyAll_forFillUpSingleClass['ifWoIn'] == 1]['pairTomoID'].values 
        tailRiboInfo[:,2:5] =  particleSt["p1"]["positions"][tailRiboIdx, ]   
        headRiboInfo[:,2:5] =  particleSt["p1"]["positions"][headRiboIdx, ]
        tailRiboInfo[:,5:] =   particleSt["p1"]["angles"][tailRiboIdx, ]   
        headRiboInfo[:,5:] =   particleSt["p1"]["angles"][headRiboIdx, ]  
        #for each tail and each head, just add interpoltaeN particles       
        for singleTail in range(tailRiboInfo.shape[0]):
            idx = int(tailRiboInfo[singleTail,0])
            example_info = particleData.iloc[idx,:]
            pos =  tailRiboInfo[singleTail, 2:5]
            ang = tailRiboInfo[singleTail, 5:]
            appendRiboStruct = interplotRibo(pos,ang,avgShift,avgRot,interpoltaN, idx, example_info, particleSt, 'tail',100)
            interplotRiboStruct = pd.concat([interplotRiboStruct, appendRiboStruct], axis = 0)
        for singleHead in range(headRiboInfo.shape[0]):
            idx = int(headRiboInfo[singleHead, 0])
            example_info = particleData.iloc[idx,:]
            pos =  headRiboInfo[singleHead, 2:5]
            ang = headRiboInfo[singleHead, 5:]
            appendRiboStruct = interplotRibo(pos,ang,avgShift,avgRot,interpoltaN, idx, example_info, particleSt, 'head',100)            
            interplotRiboStruct = pd.concat([interplotRiboStruct, appendRiboStruct], axis = 0)
        
    allInterplotRibo = generateStarInfos()
    allInterplotRibo['data_particles'] = interplotRiboStruct
    tom_starwrite(particleOut, allInterplotRibo)
    return allInterplotRibo
    
def interplotRibo(pos,ang,avgShift,avgRot, interpoltaN, idx, example_info, particleSt, flag, pruneRad,xyzborder = None):
    coordsList = np.array([]).reshape(-1, 3)
    anglesList = np.array([]).reshape(-1, 3)

    if flag == 'tail':       
        ang1 = ang
        pos1 = pos
        compare_array = np.zeros([2,3])
        compare_array[0,:] = avgRot
        compare_array[1,:] = ang1
        #calculate euler angles of relative rotation
        ang2, _, _ = tom_sum_rotation(compare_array, np.zeros([2,3]))
        pos2 = tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) + pos1           
        #if exist?
        if checkRibo(particleSt,idx, pos2, pruneRad) == 1:
            return None
        
        #out of border?
        if xyzborder is not None:
            if np.sum(pos2 > xyzborder) < 0:
                coordsList = np.concatenate((coordsList, np.array([[pos2[0],pos2[1],pos2[2]]])), axis = 0)
                anglesList = np.concatenate((anglesList, np.array([[ang2[0],ang2[1],ang2[2]]])), axis = 0)
        else:
            
            coordsList = np.concatenate((coordsList, np.array([[pos2[0],pos2[1],pos2[2]]])), axis = 0)
            anglesList = np.concatenate((anglesList, np.array([[ang2[0],ang2[1],ang2[2]]])), axis = 0)            
        cycles = interpoltaN - 1        
        #add more than one ribosomes at end of each polysome
        while cycles > 0:   
            ang1 = ang2
            pos1 = pos2
            compare_array[1,:] = ang1
            ang2, _, _ =  tom_sum_rotation(compare_array, np.zeros([2,3]))
            pos2 = tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) + pos1
            #already in the detection?
            if checkRibo(particleSt,idx, pos2,pruneRad) == 1:
                appendRiboStruct = updateParticle(coordsList, anglesList, example_info, example_info['rlnMicrographName'], -1, 'relion2')
                return  appendRiboStruct           
            #out of the border?
            if xyzborder is not None:
                if np.sum(pos2 > xyzborder) < 0:
                    coordsList = np.concatenate((coordsList, np.array([[pos2[0],pos2[1],pos2[2]]])), axis = 0)
                    anglesList = np.concatenate((anglesList, np.array([[ang2[0],ang2[1],ang2[2]]])), axis = 0)
                    
            else:
                coordsList = np.concatenate((coordsList, np.array([[pos2[0],pos2[1],pos2[2]]])), axis = 0)
                anglesList = np.concatenate((anglesList, np.array([[ang2[0],ang2[1],ang2[2]]])), axis = 0)    
            
            cycles = cycles - 1
            
        #generate the addparticle Data 
        appendRiboStruct = updateParticle(coordsList, anglesList, example_info, example_info['rlnMicrographName'], -1, 'relion2')
        return appendRiboStruct
        
        
    else:
        ang2 = ang
        pos2 = pos
        compare_array = np.zeros([2,3])
        compare_array[0,:] = np.array([-ang2[1],-ang2[0],-ang2[2]])
        compare_array[1,:] = avgRot
        #calculate euler angles of relative rotation
        ang1, _, _ = tom_sum_rotation(compare_array, np.zeros([2,3]))
        ang1 = np.array([-ang1[1],-ang1[0],-ang1[2]])
        pos1 = pos2 - tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) 
        #if exist?
        if checkRibo(particleSt,idx, pos1, pruneRad) == 1:
            return None
        
        #out of border?
        if xyzborder is not None:
            if np.sum(pos2 > xyzborder) < 0:
                coordsList = np.concatenate((coordsList, np.array([[pos1[0],pos1[1],pos1[2]]])), axis = 0)
                anglesList = np.concatenate((anglesList, np.array([[ang1[0],ang1[1],ang1[2]]])), axis = 0)
        else:
            coordsList = np.concatenate((coordsList, np.array([[pos1[0],pos1[1],pos1[2]]])), axis = 0)
            anglesList = np.concatenate((anglesList, np.array([[ang1[0],ang1[1],ang1[2]]])), axis = 0)            
        cycles = interpoltaN - 1        
        #add more than one ribosomes at end of each polysome
        while cycles > 0:  
            ang2 = ang1
            pos2 = pos1
            compare_array[0,:] = np.array([-ang2[1],-ang2[0],-ang2[2]])
            ang1, _, _ =  tom_sum_rotation(compare_array, np.zeros([2,3]))
            ang1 = np.array([-ang1[1],-ang1[0],-ang1[2]])
            pos1 = pos2 - tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) 
            #already in the detection?
            if checkRibo(particleSt,idx, pos2,pruneRad) == 1:
                appendRiboStruct = updateParticle(coordsList, anglesList, example_info, example_info['rlnMicrographName'], -1, 'relion2')
                return  appendRiboStruct           
            #out of the border?
            if xyzborder is not None:
                if np.sum(pos2 > xyzborder) < 0:
                    coordsList = np.concatenate((coordsList, np.array([[pos1[0],pos1[1],pos1[2]]])), axis = 0)
                    anglesList = np.concatenate((anglesList, np.array([[ang1[0],ang1[1],ang1[2]]])), axis = 0)
                    
            else:
                coordsList = np.concatenate((coordsList, np.array([[pos1[0],pos1[1],pos1[2]]])), axis = 0)
                anglesList = np.concatenate((anglesList, np.array([[ang1[0],ang1[1],ang1[2]]])), axis = 0)    
            
            cycles = cycles - 1
            
        #generate the addparticle Data 
        appendRiboStruct = updateParticle(coordsList, anglesList, example_info, example_info['rlnMicrographName'], -1, 'relion2')
        return appendRiboStruct 