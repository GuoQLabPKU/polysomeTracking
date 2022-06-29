import os
import numpy as np

from nemotoc.py_io.tom_starread import tom_starread
from nemotoc.py_io.tom_extractData import tom_extractData
from nemotoc.py_io.tom_starwrite import tom_starwrite
from nemotoc.py_transform.tom_average_rotations import tom_average_rotations
from nemotoc.py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp

def tom_genListFromTransForm(transList, outputFolder, polyType = 'center', listFlav = 'rel'):
    '''
    
    TOM_GENLISTFROMTRANSFORM generate list form transform paris (e.g. relion .start file or .coords)

    transListSel= tom_selectTransFormClasses(transList,selList,outputFolder)

    PARAMETERS

    INPUT
      transList                 transformation List
      outputFolder              folder for output
      posType                   ('center') center of pair
                                          'particle'  as center
      listFlav                  ('rel') flavour for output 
                                        rel for relion  


    OUTPUT
      tom_genListFromTransForm(transList,'class1','center');
      tom_genListFromTransForm(transList,'class1','particle','rel');


    '''
    
    if isinstance(transList, str):
        transList = tom_starread(transList)
    st = tom_extractData(transList)
    os.mkdir(outputFolder)
    
    if polyType == 'center':
        baseNameOut = "%s/pairCenter"%outputFolder
        ext = '.coords'
        writePairCenterCoords(st, baseNameOut, ext)
    if polyType == 'particle':
        writeParticleCenterList(st, listFlav, outputFolder)
        
    

def writePairCenterCoords(st, baseNameOut, ext):  #the transList from one class, and also analysis each tomo
                                                  #therefore, one tomogram and also from one class of transForm
    allTomoNames = st['label']['tomoName']
    allTomoNamesU = np.unique(allTomoNames)
    
    for single_tomo in allTomoNamesU:
        _, fileName = os.path.split(single_tomo)
        name, _ = os.path.splitext(fileName)
        fileNameOut = "%s_%s%s"%(baseNameOut, name, ext)
        fileNameOutCmd = "%s_%s%s.angles.zyz"%(baseNameOut, name, ext)
        
        fid = open(fileNameOut, 'w')
        fid2 = open(fileNameOutCmd, 'w')
        idx = np.where(allTomoNames == single_tomo)[0]
        positionsP1 = st['p1']['positions'][idx,:] 
        positionsP2 = st['p2']['positions'][idx,:]
        anglesP1 = st['p1']['angles'][idx,:]
        anglesP2 = st['p2']['angles'][idx,:]
        
        for i in range(positionsP1.shape[0]):
            p1 = positionsP1[i,:]; p2 = positionsP2[i,:]
            pM = np.round((p1+p2)/2)
            print('%d %d %d'%(pM[0], pM[1], pM[2]), file = fid)
            
            a1 = anglesP1[i,:].reshape(1,-1)
            a2 = anglesP2[i,:].reshape(1,-1)
            aM,_,_ = tom_average_rotations(np.concatenate((a1,a2), axis = 0))
            
            _, aMZYZ = tom_eulerconvert_xmipp(aM[0], aM[1], aM[2], 'tom2xmipp')
            print('%d %d %d %f %f %f'%(pM[0], pM[1], pM[2], aMZYZ[0], aMZYZ[1], aMZYZ[2]), file = fid2)
            
        fid.close() 
        fid2.close()
            
def writeParticleCenterList(st, listFlav, outFold):
    starOrg = tom_starread(st['label']['orgListName'][0])
    starOrgData = starOrg['data_particles']
   
    #store the coordinates of all
    all_ind = np.concatenate((st['p1']['orgListIDX'], st['p2']['orgListIDX']))
    all_ind = np.unique(all_ind)   
    starNew = starOrgData.iloc[all_ind, :]
    starOrg['data_particles'] = starNew 
    tom_starwrite('%s/allParticles.star'%outFold, starOrg)