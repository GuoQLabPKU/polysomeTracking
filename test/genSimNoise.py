import numpy as np
import pandas as pd
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_starread import generateStarInfos

def genNoise(eulerAngles = None):
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
        
    #generate the noise conf
    conf = []
    zz = { }
    zz['tomoName'] = '100.mrc'
    zz['numRepeats'] = 5000
    zz['minDist'] = 50
    zz['searchRad'] = 100
    conf.append(zz)
        
    #code 
    for single_conf in conf:
        list_ = addNoisePoints(single_conf['tomoName'],
                               single_conf['numRepeats'], single_conf['minDist'],
                               single_conf['searchRad'], eulerAngles)
    
   
    starInfo = generateStarInfos()
    starInfo['data_particles'] = list_
    tom_starwrite('simNoise.star', starInfo) 
    
          
def addNoisePoints(tomoName, nrNoisePoints, minDist, searchRad, eulerAngles):
    listNoise = allocListFrame(tomoName, nrNoisePoints, 'relion')  
    if eulerAngles is not None:
        eulerAnglesLen = eulerAngles.shape[0]
    half_noiseN = int(listNoise.shape[0]/2)

    for i in range(half_noiseN):     
        posNoise = getPositionsPerTomo(listNoise, tomoName)
        pos = genUniquePos(posNoise,  minDist)
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
    for i in range(half_noiseN, nrNoisePoints):
        posNoise = getPositionsPerTomo(listNoise, tomoName)        
        pos = genUniquePos_adjact(posNoise, posListNoise, minDist, searchRad)[0]
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
        
   

    listNoise.reset_index(inplace = True, drop = True)
    
    return listNoise
        
def genUniquePos_adjact(oldPos, posSeed, minDist, offSize):
    for i in range(1000):      
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

def genUniquePos(oldPos, minDist):
    xrange = (-1000, 1000)
    yrange = (-1000, 1000)
    zrange = (-1000, 1000)
    for i in range(1000):
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
                       'rlnAngleRot':rlnAngleRot, 'rlnAngleTilt':rlnAngleTilt,
                       'rlnImageName':rlnImageName, 'rlnMagnification':rlnMagnification,
                       'rlnDetectorPixelSize':rlnDetectorPixelSize, 'rlnCtfImage':rlnCtfImage,
                       'rlnGroupNumber':rlnGroupNumber, 'rlnOriginX':rlnOriginX,
                       'rlnOriginY':rlnOriginY, 'rlnOriginZ':rlnOriginZ,
                       'rlnClassNumber':rlnClassNumber, 'rlnNormCorrection':rlnNormCorrection,
                       'rlnLogLikeliContribution':rlnLogLikeliContribution, 
                       'rlnMaxValueProbDistribution':rlnMaxValueProbDistribution, 
                       'rlnNrOfSignificantSamples':rlnNrOfSignificantSamples})
    return st