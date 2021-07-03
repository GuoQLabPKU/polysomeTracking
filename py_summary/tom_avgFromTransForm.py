import os 
import numpy as np
import subprocess
import re

def tom_avgFromTransForm(transList, maxRes, pixS, classFilt = -1, outputRoot = 'avg/r1', avgCall = 'tom'):
    '''
    TOM_AVGFROMTRANSFORM generate avg volumes form transform paris 

    tom_avgFromTransForm(transList,classFilt,outputFolder)

    PARAMETERS

    INPUT
        transList                        transformation List or wildCard to nativeLists
        classFilt                        (-1)structure or class number
        outputRoot                       (avg/r1) folder for output
        maxRes                           max res in Ang 
        pixS                             (1) pixesize of particles
        avgCall                          ('tom') call for avg generation   

    OUTPUT
        avg                              avg volume

    EXAMPLE
    avgCall='mpirun -np 35 relion_reconstruct_mpi --i XXX_inList_XXX --maxres XXX_maxRes_XXX --ctf --3d_rot --o XXX_outAVG_XXX';
    tom_avgFromTransForm('myPolysomes342/run4/classes/c*/particleCenter/allPart.star',-1,'out/r0',35,avgCall);

    ''' 
    if type(transList).__name__ == 'str':
        isPairTransForm = os.path.exists(transList)
    else:
        isPairTransForm = 1
        
    #make dir 
    if not os.path.exists(os.path.split(outputRoot)[0]):
        os.mkdir(os.path.split(outputRoot)[0])
    if not os.path.exists('%s/log'%os.path.split(outputRoot)[0]):
        os.mkdir('%s/log'%os.path.split(outputRoot)[0])
    if isPairTransForm:
        pass
    else:
        avgFromWildCard(transList, outputRoot, classFilt, avgCall, maxRes, pixS)
    
def avgFromWildCard(wk, outputRoot, classFilt, avgCallTmpl, maxRes, pixS):
    #list the dir for the whole classes
    wk_upup = os.path.split(os.path.split( os.path.split(wk)[0]) [0])[0]
    d = [ i+ '/particleCenter/allParticles.star' for i in os.listdir(wk_upup)]     
    tmp = os.path.split(os.path.split(os.path.split(wk)[0])[0])[1]
    
    if isinstance(classFilt, int) | isinstance(classFilt, float):
        maxLen = np.inf
    else:
        if 'maxNumPart' in classFilt.keys():
            maxLen = classFilt['maxNumPart']
        else:
            maxLen = classFilt['maxNumTransForm']
    
    idx = [ ]
    
    if type(classFilt).__name__ == 'dict':
        classes = np.zeros(len(d), dtype = np.int)
        lens = np.zeros(len(d), dtype = np.int)
        for i in len(d):
            inputName = "%s/%s"%(wk_upup, d[i])
            #get the up-dir 
            classes[i] = int(tmp.replace('c',''))
            if 'maxNumPart' in classFilt.keys():
                call = 'cat %s  | awk \"NF>5{print $0}\" | wc -l'%inputName
            else:
                inputNameTR = inputName.replace('/particleCenter/allParticles.star', '/transList.star')
                call = 'cat %s  | awk \"NF>5{print $0}\" | wc -l '%inputNameTR
            
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            out, err = p.communicate()
            
            res = [int(i) for i in re.findall('\d+', out)][0]
            lens[i] = res
       
            if res > classFilt['minNumTransform']: ##select the classes which has more than transforms 
                idx.append(i)
                
    if len(idx) == 0:
        return 
    for i in range(len(idx)):
        uPos = idx[i]
        inputName = "%s/%s"%(wk_upup, d[uPos])
        #fold = os.path.split(inputName)[0]
        p = os.path.split(os.path.split(os.path.split(inputName)[0])[0])[1]
        outputName = '%s/%s.mrc'%(outputRoot, p)
        outputNameLog = '%s/log/%s.log'%(os.path.split(outputRoot)[0], p)
        
        if lens(uPos) > maxLen:  #Iguess last section already did this 
            inputNameTmp = inputName.replace('.star', '_subset.star')
            call = 'cat %s | awk \"NF<4{print $0}\" > %s'%(inputName, inputNameTmp)
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            call = 'awk \"NF>5{print $0}\" %s | sort -R | awk \"NR<%d{print $0}\" >> %s'%(inputName, 
                              maxLen, inputNameTmp)
            p = subprocess.Popen(call, shell = True, stdout = subprocess.PIPE)
            inputName = inputNameTmp   ##if too many transforms, only keep maxlen transforms/particles
        
        if type(avgCallTmpl).__name__ == 'dict':
            print("############################################################")
            print("exect below commands in the matlab")
            command = "tom_sg_average(%s,%s,%s,%s,\'\', \'default\', %d, %3f"%(inputName, 
            avgCallTmpl['subtomoPath'],avgCallTmpl['invert'], outputName, maxRes, pixS)                                                                   
            print(command)
            print("############################################################")
        else:
            avgCall = avgCallTmpl.replace('XXX_inList_XXX', inputName)
            avgCall = avgCall.replace('XXX_outAVG_XXX', outputName)
            avgCall = avgCall.replace('XXX_maxRes_XXX', str(maxRes))
            avgCallFull = "%s &> %s"%(avgCall, outputNameLog)
            p = subprocess.Popen(avgCallFull, shell = True, stdout = subprocess.PIPE)
        
        #the idea is give a number cutoff of transforms for each class, and then average 
        #the transform for each class!
     
            
            
            
        
        
            
            
            
            
            
            
    
    
        
