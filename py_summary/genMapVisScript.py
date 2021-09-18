import os 
import numpy as np


from py_io.io_tomo import load_tomo 
from py_io.tom_starread import tom_starread

def genMapVisScript(mapfold, classFile, scriptName, transList, voxelSize, offSet):
    
    mrc_file = [i for i in os.listdir(mapfold) if '.mrc' in i]
    mrc_abspath = ["%s/%s"%(mapfold,i) for i in mrc_file ]
    if len(mrc_file) == 0:
        print('warning no models in %s *.mrc, ... check your filter setting'%mapfold)
        print('skipping')
    
    sz = load_tomo(mrc_abspath[0])
    sz = sz.shape[0]
    pos = calcGrd(len(mrc_abspath), sz)  #len(mrc_abspath)*3 2D-array
    
    clInfo = tom_starread(classFile)
    clInfo = clInfo['data_particles']
    allClnfo = clInfo['classNr'].values
    allCl = transList['pairClass'].values
    
    if os.path.exists(scriptName):
        os.remove(scriptName)
    allClN = np.zeros(len(mrc_abspath), dtype = np.int)  #1D array
    zz = -1
    
    for i in range(len(mrc_file)):
        b = os.path.splitext(i)[0]
        b = b.split('.')[0].replace('c','')
        try:
            clN = int(b)
        except ValueError:
            print('error:could not parse number from: %s skipping'%i)
        
        zz += 1
        allClN[zz] = clN
        idx = np.where(allCl == clN)[0]
        if len(idx) > 0:
            col = transList['pairClassColour'][idx[0]]
            idx2 = np.where(allClnfo == clN)[0]  #why analsis everthing for class 0?
            infoStr = 'clNr: %d num: %d  poly>5: %d'%(clN, clInfo['num'].values[idx2[0]], clInfo['numPolybg5'].values[idx2[0]] )
            pathToMap = mrc_abspath[i]
            writeChimeraCmd(scriptName, pathToMap, col, clN, voxelSize, offSet, pos[i,:], infoStr,sz,i)
            
    writeChimeraTail(scriptName, max(allClN) + 1)



def writeChimeraTail(scriptName, maxCl):
    fid = open(scriptName, 'a')
    markerName = scriptName.replace('.cmd', '_marker.cmm')
    cmd = 'sop hideDust #0-1000 size 20'
    print(cmd, file = fid)
    cmd = 'volume all step 1'
    print(cmd, file = fid)
    
    cmd = 'open %s'%markerName
    print(cmd, file = fid)
    cmd = 'focus #0-200'
    print(cmd, file = fid)
    
    fid.close()
    fid = open(markerName, 'a')
    print('</marker_set>', file = fid)
    fid.close()
                
        
def  writeChimeraCmd(scriptName, pathToMap, col, clN, voxelSize, offSet, pos, infoStr, sz, mNr):
    fid = open(scriptName,'a')
    cmd = 'open %d  %s'%(clN, pathToMap)
    print(cmd, file = fid)
    
    cmd = 'volume #%d voxelSize %d'%(clN, voxelSize)
    print(cmd, file = fid)
    
    offTot = pos + np.repeat(offSet, 3) # 1D array with 3 elements
    cmd = 'volume #%d originIndex %d, %d, %d'%(clN, offTot[0], offTot[1], offTot[2])
    print(cmd, file = fid)
    
    cmd = 'volume #%d color %s'%(clN, '-'.join([str(i) for i in col]))
    print(cmd, file = fid)
    
    markerName = scriptName.replace('.cmd', '_marker.cmm')
    if mNr == 0:
        fid = open(scriptName.replace('.cmd',  '_marker.cmm'  ), 'w')
        print('<marker_set name=\"marker set avg info\">',file = fid)
    else:
        fid = open(markerName, 'a')
    offFromCorner = np.array([0,0, np.round(sz/2)*voxelSize])
    posAbs = pos*-1*voxelSize
    posAbs = posAbs + offFromCorner
    
    cmd = "<marker id=\"%d\" x=\"%.2f\" y=\"%.2f\" z=\"%.2f\" radius=\"0.1\" note=\"%s\">"%(mNr, posAbs[0], posAbs[1], posAbs[2],infoStr)
    print(cmd, file = fid)
    fid.close()
    

def calcGrd(nr, szVol):
    numLen = np.ceil(np.sqrt(nr))
    zz = -1
    pos = np.zeros((nr, 3), dtype = np.int)
    
    for i in range(numLen):
        for ii in range(numLen):
            zz += 1
            pos[zz, :] = [i*szVol, ii*szVol, 0]
            if (zz+1) >= nr:
                break
            
        if zz>=nr:
             break
    return pos