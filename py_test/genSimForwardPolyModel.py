import numpy as np
import pandas as pd
from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_starread import generateStarInfos

def genForwardPolyModel(conf = None, eulerAngles=None, save_flag = ''):
    '''
    Parameters
    ----------
    conf : dict, optional
        the information to generate polysomes. The default is None.

    Returns
    -------
    simulated star file and store a npy file of polysomes

    '''
    if isinstance(eulerAngles, str):
        print('pick euler angles from %s'%eulerAngles)
        eulerAngles = pd.read_csv(eulerAngles, sep=",")
    if conf == None:
        conf = []
        zz = {}
        zz['type'] = 'vect'
        zz['tomoName'] = '100.mrc'
        zz['numRepeats'] = 30 #the length of the polysome
        zz['increPos'] = np.array([60,4,10])
        zz['increAng'] = np.array([10,20,30])
        zz['startPos'] = np.array([500,0,0])
        zz['startAng'] = np.array([-20,-10,-30])
        zz['minDist'] = 50
        zz['searchRad'] = 100
        zz['branch'] = 0
        zz['noizeDregree'] = 2
        conf.append(zz)
        zz2 = { }
        zz2['type'] = 'noise'
        zz2['tomoName'] = '100.mrc'
        zz2['numRepeats'] = 200
        zz2['minDist'] = 50
        zz2['searchRad'] = 100
        conf.append(zz2)
        
    #code 
    idxBranches = [ ]
    list_ = 'init'
    polysome_flag = dict()
    polysome_Nr = 1
    for single_conf in conf:
        if single_conf['type'] == 'vect':
            if isinstance(list_, str):
                shape0 = 0
            else:      
                shape0 = list_.shape[0]
            list_,  idx_branch = genVects(list_, single_conf['tomoName'], single_conf['increPos'],
                              single_conf['increAng'], single_conf['startPos'],
                              single_conf['startAng'], single_conf['numRepeats'], 
                              single_conf['branch'], single_conf['noizeDregree'],single_conf['searchRad'])
            shape1 = list_.shape[0]
            polysome_flag[polysome_Nr] = (shape0, shape1)
            polysome_Nr += 1
            if single_conf['branch']:
                idxBranches.append(idx_branch)
        if single_conf['type'] == 'noise':
            list_ = addNoisePoints(list_, single_conf['tomoName'],
                                   single_conf['numRepeats'], single_conf['minDist'],
                                   single_conf['searchRad'], eulerAngles)
    
    polysome_label = np.zeros(list_.shape[0], dtype = np.int)#label the polysome information to the data 
    for key in polysome_flag.keys():
        begin, end = polysome_flag[key]
        polysome_label[begin:end] = key
    list_['polysome'] = polysome_label
    #print(list_['polysome'])
    writeStarFile(list_, 0,save_flag)
    
    return idxBranches
def writeStarFile(list_, save_npy = 1, save_flag = ''):
    
    starInfo = generateStarInfos()
    starInfo['data_particles'] = list_
    if len(save_flag) == 0:
        tom_starwrite('sim.star', starInfo)
    else:   
        tom_starwrite('sim_%s.star'%save_flag, starInfo) 
    

    id_ = np.random.permutation(list_.shape[0])
    list_ = list_.loc[id_,:]
    list_.reset_index(drop=True,inplace = True)  
    #save the polysome information
    ori_polysome = dict()
    polysome_unique = np.unique(list_['polysome'].values)
    for single_polysome in polysome_unique:
        if single_polysome == 0:
            continue
        idx = list_[list_['polysome'] == single_polysome].index
        ori_polysome[np.min(idx)] = set(idx)
    #save the dict 
    if save_npy:
        np.save('./py_test/ori_polysome.npy', ori_polysome)      
    list_.drop('polysome',axis = 1,inplace = True)
    starInfo['data_particles'] = list_
    if len(save_flag) == 0:
        tom_starwrite('simOrderRandomized.star',starInfo)
    else:             
        tom_starwrite('simOrderRandomized_%s.star'%save_flag, starInfo)
    
def genVects(list_, tomoName, increPos, increAng, startPos, startAng, nrRep, branch=0,
             noizeDregree = 2, searchRad = 100):
    if not isinstance(list_ ,str):
        listIn = list_
    else:
        listIn = pd.DataFrame({})
    list_ = allocListFrame(tomoName, nrRep, 'relion')
    
    posOld = startPos
    angOld = startAng
    
    for i in range(nrRep):
        if branch:
            angNoise = np.zeros(3)
            
        else:   
            angNoise = np.random.rand(3)*noizeDregree  
                      
        ang,_,_ = tom_sum_rotation(np.array([list(increAng), list(angOld), list(angNoise)]),
                                         np.zeros((3,3)))
        
        #make sure the distance is smaller than the search radius
        vTr = np.array([10000,10000,10000])
        counts = 0
        while (np.linalg.norm(vTr) >= searchRad) & (counts < 1000):
            if branch:
                vnoise = np.zeros(3)
            else:
                vnoise = np.random.rand(3)*noizeDregree

            vTr = tom_pointrotate(increPos + vnoise, ang[0], ang[1], ang[2])
            pos = posOld + vTr
            counts += 1
        if (counts == 1000):
            print('the noise for translocation is too big, will ignore the noise!')
            vTr = tom_pointrotate(increPos, ang[0], ang[1], ang[2])
            pos = posOld + vTr
            
            
        
        
        list_.loc[i,'rlnCoordinateX'] = pos[0]
        list_.loc[i,'rlnCoordinateY'] = pos[1]
        list_.loc[i,'rlnCoordinateZ'] = pos[2]
        
        
        _, angC = tom_eulerconvert_xmipp(ang[0], ang[1], ang[2], 'tom2xmipp')
        list_.loc[i,'rlnAngleRot'] = angC[0]
        list_.loc[i,'rlnAngleTilt'] = angC[1]
        list_.loc[i,'rlnAnglePsi'] = angC[2]
        
        posOld = pos
        angOld = ang
        
    #generate branch from position5:
    if branch:
        pos = list_.loc[5, ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        ang = list_.loc[5, ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values
        _, ang = tom_eulerconvert_xmipp(ang[0], ang[1], ang[2], 'xmipp2tom')
        listBranch = genBranch(pos, ang, increAng, increPos, 6, tomoName)
        idxBranch = (listIn.shape[0] + 5, listIn.shape[0] + list_.shape[0])
        list_ = pd.concat((listIn, list_, listBranch),axis = 0)
    else:
        idxBranch = (0,0)
        list_ = pd.concat((listIn, list_),axis = 0)
        
    list_.reset_index(inplace = True, drop = True)
    
    return list_, idxBranch
        
def genBranch(pos, ang, increAng, increPos, nrRep, tomoName, noiseDregre = 3):
    
    list_ = allocListFrame(tomoName, nrRep, 'relion')
    posOld = pos
    angOld = ang  
    
    for i in range(nrRep):   
        angNoise = np.random.rand(3)*noiseDregre
        vnoise = np.random.rand(3)*noiseDregre
        ang,_,_ = tom_sum_rotation( np.array([list(increAng), list(angOld), list(angNoise)]),
                                    np.zeros((3,3)))
                  
        vTr = tom_pointrotate(increPos + vnoise, ang[0], ang[1], ang[2])
        pos = posOld + vTr
        
        list_.loc[i,'rlnCoordinateX'] = pos[0]
        list_.loc[i,'rlnCoordinateY'] = pos[1]
        list_.loc[i,'rlnCoordinateZ'] = pos[2]
        
        
        _, angC = tom_eulerconvert_xmipp(ang[0], ang[1], ang[2], 'tom2xmipp')
        list_.loc[i,'rlnAngleRot'] =  angC[0]
        list_.loc[i,'rlnAngleTilt'] = angC[1]
        list_.loc[i,'rlnAnglePsi'] =  angC[2]
        
        posOld = pos
        angOld = ang   
        
    return list_
    
       
def addNoisePoints(list_, tomoName, nrNoisePoints, minDist, searchRad, eulerAngles):
    listNoise = allocListFrame(tomoName, nrNoisePoints, 'relion')  
    if eulerAngles is not None:
        eulerAnglesLen = eulerAngles.shape[0]
    half_noiseN = int(listNoise.shape[0]/2)
    posList = getPositionsPerTomo(list_, tomoName)

    for i in range(half_noiseN):
       
        posNoise = getPositionsPerTomo(listNoise, tomoName)
        oldPos = np.concatenate((posList, posNoise), axis = 1)

        pos = genUniquePos(oldPos, posList, minDist)
        listNoise.loc[i,'rlnCoordinateX'] = pos[0]
        listNoise.loc[i,'rlnCoordinateY'] = pos[1]
        listNoise.loc[i,'rlnCoordinateZ'] = pos[2]
        
        #get random euler angles from eulerAngles
        if eulerAngles is not None:
            index_rand = np.random.choice(range(eulerAnglesLen),1)[0]                     
            angC = eulerAngles.iloc[index_rand,:].values*180/np.pi
                     
        else:    
            angC = np.random.rand(3)*360
        listNoise.loc[i,'rlnAngleRot'] = angC[0]
        listNoise.loc[i,'rlnAngleTilt'] = angC[1]
        listNoise.loc[i,'rlnAnglePsi'] = angC[2]
        
    posListNoise = posNoise[:, 0:half_noiseN-1]
    posList = np.concatenate((posList, posListNoise), axis = 1)
    for i in range(half_noiseN, nrNoisePoints):
        posNoise = getPositionsPerTomo(listNoise, tomoName)
        oldPos = np.concatenate((posList, posNoise), axis = 1)
        
        pos = genUniquePos_adjact(oldPos, posList, minDist, searchRad)[0]
        listNoise.loc[i,'rlnCoordinateX'] = pos[0]
        listNoise.loc[i,'rlnCoordinateY'] = pos[1]
        listNoise.loc[i,'rlnCoordinateZ'] = pos[2]
        #get random euler angles from eulerAngles
        if eulerAngles is not None:
            index_rand = np.random.choice(range(eulerAnglesLen),1)[0]                     
            angC = eulerAngles.iloc[index_rand,:].values*180/np.pi
            
        else:    
            angC = np.random.rand(3)*360
        listNoise.loc[i,'rlnAngleRot'] = angC[0]
        listNoise.loc[i,'rlnAngleTilt'] = angC[1]
        listNoise.loc[i,'rlnAnglePsi'] = angC[2]        
        
   
    list_ = pd.concat((list_,listNoise),axis = 0)
    list_.reset_index(inplace = True, drop = True)
    
    return list_
        
def genUniquePos_adjact(oldPos, posSeed, minDist, offSize):
    for i in range(10000):      
        ind = np.random.permutation(posSeed.shape[1])
        seed = posSeed[:,ind[0]]
        m = np.fix(np.random.rand(3) + 0.5)
        m = m*2-1
        pos = seed.reshape(1,-1) + np.random.rand(3)*offSize*m
        allDist = oldPos - np.tile(pos, (oldPos.shape[1],1)).transpose()
        allDist = np.sqrt(np.sum(allDist*allDist,axis = 0))
        allDist = allDist<minDist
        if np.sum(allDist) < 1:
            break      
    return pos

def genUniquePos(oldPos, posSeed, minDist):
    posSeed=posSeed[:, posSeed[0,:] > -1000]
    xrange = (int(np.min(posSeed[0,:])), int(np.max(posSeed[0,:])))
    yrange = (int(np.min(posSeed[1,:])), int(np.max(posSeed[1,:])))
    zrange = (int(np.min(posSeed[2,:])), int(np.max(posSeed[2,:])))
    for i in range(10000):
        pos = np.array([np.random.choice(range(xrange[0], xrange[1]),1)[0],
                        np.random.choice(range(yrange[0], yrange[1]),1)[0],
                        np.random.choice(range(zrange[0], zrange[1]),1)[0]])
        
        allDist = oldPos - np.tile(pos, (oldPos.shape[1],1)).transpose()
        allDist = np.sqrt(np.sum(allDist*allDist, axis = 0))
        allDist = allDist < minDist
        if np.sum(allDist) < 1:
            break
    return pos
        
def getPositionsPerTomo(list_, tomoName):
    allNames = list_['rlnMicrographName'].values
    idx = np.where(allNames == tomoName)[0]
    pos = np.zeros((3,len(idx)))
    
    for i in range(len(idx)):
        ind = idx[i]
        pos[:,i] = np.array([list_['rlnCoordinateX'].values[ind],
                             list_['rlnCoordinateY'].values[ind],
                             list_['rlnCoordinateZ'].values[ind]])
    return pos
    
def allocListFrame(tomoName, nrItem, flavour):
    rlnCoordinateX = [ ]
    rlnCoordinateY = [ ]
    rlnCoordinateZ = [ ]
    rlnMicrographName = [ ]
    rlnAngleRot = [ ]
    rlnAngleTilt = [ ]
    rlnAnglePsi = [ ]
    rlnImageName = [ ]
    rlnMagnification = [ ]
    rlnDetectorPixelSize = [ ]
    rlnCtfImage = [ ]
    rlnGroupNumber = [ ]
    rlnOriginX = [ ]
    rlnOriginY = [ ]
    rlnOriginZ = [ ]
    rlnClassNumber = [ ]
    rlnNormCorrection = [ ]
    rlnLogLikeliContribution = [ ]
    rlnMaxValueProbDistribution = [ ]
    rlnNrOfSignificantSamples = [ ]
    
    
    if flavour == 'relion':
        for i in range(nrItem):
            rlnCoordinateX.append(-10000)
            rlnCoordinateY.append(-10000)
            rlnCoordinateZ.append(-10000)
            rlnMicrographName.append(tomoName)
            rlnAngleRot.append(0)
            rlnAngleTilt.append(0)
            rlnAnglePsi.append(0)
            rlnImageName.append('notImplemented.mrc')
            rlnMagnification.append(10000)
            rlnDetectorPixelSize.append(3.42)
            rlnCtfImage.append('notImplemented.mrc')
            rlnGroupNumber.append(0)
            rlnOriginX.append(0)
            rlnOriginY.append(0)
            rlnOriginZ.append(0)
            rlnClassNumber.append(0)
            rlnNormCorrection.append(0)
            rlnLogLikeliContribution.append(0)
            rlnMaxValueProbDistribution.append(0)
            rlnNrOfSignificantSamples.append(1)
            
    st = pd.DataFrame({'rlnCoordinateX':rlnCoordinateX,'rlnCoordinateY':rlnCoordinateY,
                       'rlnCoordinateZ':rlnCoordinateZ, 'rlnMicrographName':rlnMicrographName,
                       'rlnAngleRot':rlnAngleRot, 'rlnAngleTilt':rlnAngleTilt, 'rlnAnglePsi':rlnAnglePsi,
                       'rlnImageName':rlnImageName, 'rlnMagnification':rlnMagnification,
                       'rlnDetectorPixelSize':rlnDetectorPixelSize, 'rlnCtfImage':rlnCtfImage,
                       'rlnGroupNumber':rlnGroupNumber, 'rlnOriginX':rlnOriginX,
                       'rlnOriginY':rlnOriginY, 'rlnOriginZ':rlnOriginZ,
                       'rlnClassNumber':rlnClassNumber, 'rlnNormCorrection':rlnNormCorrection,
                       'rlnLogLikeliContribution':rlnLogLikeliContribution, 
                       'rlnMaxValueProbDistribution':rlnMaxValueProbDistribution, 
                       'rlnNrOfSignificantSamples':rlnNrOfSignificantSamples})
    return st