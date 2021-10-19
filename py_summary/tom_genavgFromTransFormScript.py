import os 
import numpy as np
import subprocess
import re
import glob

from py_log.tom_logger import Log
def tom_genavgFromTransFormScript(transList, maxRes, pixS, workerNr = 35 ,
                                  classFilt = -1, callByPython = 0,
                                  outputRoot = 'avg/r1'):
    '''
    TOM_AVGFROMTRANSFORM generate avg volumes form transform paris 

    tom_genavgFromTransFormScript(transList,classFilt,outputFolder)

    PARAMETERS

    INPUT
        transList                        transformation List or wildCard to nativeLists
        classFilt                        (-1)structure or class number
        outputRoot                       (avg/r1) folder for output
        maxRes                           max res in Ang 
        pixS                             (1) pixesize of particles
        avgCall                          ('relion') call for avg generation   

    OUTPUT
        avg                              avg volume

    EXAMPLE
    avgCall='mpirun -np 35 relion_reconstruct_mpi --i XXX_inList_XXX --maxres XXX_maxRes_XXX --ctf --3d_rot --o XXX_outAVG_XXX';
    tom_genavgFromTransFormScript('myPolysomes342/run4/classes/c*/particleCenter/allPart.star',-1,'out/r0',35,avgCall);

    ''' 
    log = Log('average particles').getlog()
    avgCall = 'mpirun -np %d `which relion_reconstruct_mpi` --i XXX_inList_XXX --maxres XXX_maxRes_XXX --ctf --3d_rot --o XXX_outAVG_XXX'%workerNr
    if isinstance(transList,str):
        isPairTransForm = len(glob.glob(transList))>1
    else:
        isPairTransForm = 1
        
    #make dir 
    if not os.path.exists(os.path.split(outputRoot)[0]):
        os.mkdir(os.path.split(outputRoot)[0])
    if not os.path.exists('%s/log'%os.path.split(outputRoot)[0]):
        os.mkdir('%s/log'%os.path.split(outputRoot)[0])
    
    if not isPairTransForm:
        log.warning('No particles files detect. Skip average particles')  
        return
    else:
        avgFromWildCard(transList, outputRoot, classFilt, avgCall, maxRes, pixS,callByPython)
    
def avgFromWildCard(wk, outputRoot, classFilt, avgCallTmpl, maxRes, pixS, callByPython):
    #list the dir of all classes
    wk_upup = os.path.split(os.path.split(os.path.split(wk)[0]) [0])[0]
    d = [ i+ '/particleCenter/allParticles.star' for i in os.listdir(wk_upup)]     
  
    
    if isinstance(classFilt, int) | isinstance(classFilt, float):
        maxLen = np.inf
    else:
        if 'maxNumPart' in classFilt.keys():
            maxLen = classFilt['maxNumPart']
        else:
            maxLen = classFilt['maxNumTransForm']
    
    idx = [ ]
    
    if isinstance(classFilt, dict):
        lens = np.zeros(len(d), dtype = np.int)
        for i in range(len(d)):
            inputName = "%s/%s"%(wk_upup, d[i]) #each transList from one class
            #get the up-dir 
            if 'maxNumPart' in classFilt.keys():
                call = 'cat %s  | awk \"NF>5{print $0}\" | wc -l'%inputName
            else:
                inputNameTR = inputName.replace('/particleCenter/allParticles.star', '/transList.star')
                call = 'cat %s  | awk \"NF>5{print $0}\" | wc -l '%inputNameTR
            
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            out, err = p.communicate()
            
            res = [int(i) for i in re.findall('\d+', out)][0]
            lens[i] = res
            
            if 'maxNumPart' in classFilt.keys():
                if res > classFilt['minNumPart']: ##select the classes which has more than particles
                    idx.append(i)
                
            else:
                if res > classFilt['minNumTransform']:   ##select the classes which has more than transforms
                    idx.append(i)            
                
    if len(idx) == 0:
        return 
    for i in range(len(idx)):
        uPos = idx[i]
        inputName = "%s/%s"%(wk_upup, d[uPos])
        #fold = os.path.split(inputName)[0]
        p = os.path.split(os.path.split(os.path.split(inputName)[0])[0])[1]
        outputName = '%s/%s.mrc'%(os.path.split(outputRoot)[0], p)
        outputNameLog = '%s/log/%s.log'%(os.path.split(outputRoot)[0], p)
        
        #tell user call by unix or python
        if not callByPython:
            scriptName = "%s/avg.cmd"%(os.path.split(outputRoot)[0])
            print('the average script by relion is located at %d'%scriptName)
            
        if lens(uPos) > maxLen:  
            inputNameTmp = inputName.replace('.star', '_subset.star')
            call = 'cat %s | awk \"NF<4{print $0}\" > %s'%(inputName, inputNameTmp)
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            call = 'awk \"NF>5{print $0}\" %s | sort -R | awk \"NR<%d{print $0}\" >> %s'%(inputName, 
                              maxLen, inputNameTmp)
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            inputName = inputNameTmp   ##if too many transforms, only keep maxlen transforms/particles
        
            
            
            #process by relion
            avgCall = avgCallTmpl.replace('XXX_inList_XXX', inputName)
            avgCall = avgCall.replace('XXX_outAVG_XXX', outputName)
            avgCall = avgCall.replace('XXX_maxRes_XXX', str(maxRes))
            avgCallFull = "%s &> %s"%(avgCall, outputNameLog)
            if callByPython:
                p = subprocess.Popen(avgCallFull, shell = True, stdout = subprocess.PIPE)
            else:
                f = open(scriptName, 'a+')
                f.write(avgCallFull + '\n')
                f.close()
                
                