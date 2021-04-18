from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_extractData import tom_extractData

import numpy as np
import os

def tom_selectTransFormClasses(transList, selList, minNumTransForms = -1,outputFolder = '', ):
    '''
    TOM_SELECTTRANSFORMCLASSES selects classes from transForm list

    transListSel= tom_selectTransFormClasses(transList,selList,outputFolder,minNumPart)

    PARAMETERS

    INPUT
       transList                     transformation List #should be a dataframe
       selList                       selecton List #should be the property of the class
       outputFolder                  folder for output
       minNumTransForms              (-1) minimum number of transformations per class
                                     #the number of transforms of the ribosome pairs
    OUTPUT
       transListSel                  seleted transforms lists
       selFolder                     folder where selected lists have been written 
       transListSelCmb               select list combined
    
    EXAMPLE
    
       #configure 
       selList(1).classNr=[1 2];
       selList(1).polyNr=-1;
    
       #run
       tom_selectTransFormClasses(transList,selList,'run1')
    
    
    REFERENCES
    '''
    #DO NOT Change the changable data struct in function/anywhere
    type_list = type(transList).__name__
    if type_list == 'str':
        transList = tom_starread(transList)
    st = tom_extractData(transList)#st should be a dict 
    if type(selList).__name__ ==  'str':
        if selList == 'Classes-Sep':
            uClass = np.unique(st["label"]['pairClass'])  # select all the classes and ploys
            del selList
            selList = [ ]
            for i in range(len(uClass)):
                selList.append({ })
                selList[i]["classNr"] = np.array([uClass[i]]) # #can be -1 OR [0,1,2...]
                selList[i]["polyNr"] = np.array([-1]) #selList is a list with dicts stored
        else:
            raise TypeError('some unrecongnized input')


    
    idxCmb = [ ]
    transListSel = [ ]
    selFolders = [ ]
    for i in range(len(selList)):#process each class, also class0:fail to cluster
        clNr = selList[i]["classNr"]
        polyNr = selList[i]['polyNr']
        if not all(clNr):
            continue
        if clNr[0] == -1:
            continue
        idx = filterList(st, clNr, polyNr) #which class and in this class,which polysome you like?
        if len(idx) < minNumTransForms:
            continue #this class has few transforms 
        transListSel.append( transList.iloc[idx,:])
        trFilt = transList.iloc[idx,:]
        selFolders.append(genOutput(trFilt, selList[i], outputFolder)) #contain -1 when don't save class subset
        idxCmb = np.concatenate((idxCmb, idx), axis = 0)
        
    if len(idxCmb) > 0:
        idxCmb.astype(np.int)
        transListSelCmb = transList.iloc[idxCmb,:]
        #reindex the index 
        transListSelCmb.reset_index(drop = True, inplace = True)
        
        
    else:
        transListSelCmb = ''  #should save space than []
        
    return   transListSel,selFolders,transListSelCmb




def genOutput(transList, selList, outputFolder):
    if outputFolder == '':
        selFolder = ''
        return selFolder
    clNr = selList['classNr']
    polyNr = selList['polyNr']
    
    
  
    if polyNr[0] == -1:  #no polysome tracked
        polyNrStr = ''
    else:
        polyNrStr = '+'.join([str(i) for i in polyNr]) #in any case, there  only less than one '+'   
        polyNrStr = 'p%s'%polyNrStr
          
    clNrStr = '+'.join([str(i) for i in clNr]) #in any case, there  only less than one '+'
    selFolder = '%s/c%s%s'%(outputFolder, clNrStr, polyNrStr)
    os.mkdir(selFolder)
    #write starfile
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"]  = ["_%s"%i for i in transList.columns]
    tom_starwrite('%s/transList.star'%selFolder, transList, header)
    return selFolder
 
    


def filterList(st, classNr, polyNr):
    if 'pairClass' in st["label"].keys():
        if classNr[0] == -1:  #-1 represents no clustering process performed!
            idxC = np.arange(len(st['label']['pairClass']))
        else:
            allClasses = st["label"]["pairClass"]
            #judge the type of input:
            idxC = np.where(allClasses==classNr[:,None])[-1] #in case the classNr is an container
                
    else:
        idxC = np.arange(st['p1']['positions'].shape[0])
    if 'pairLabel' in st["label"].keys():
        if polyNr[0] == -1:
            idxP = np.arange(len(st['label']['pairClass']))
        else:               
            allPoly = st["label"]["pairlabel"]
            idxP = np.where(allPoly == polyNr[:,None])[-1] #in case the classNr is an container            
   
    else:
        idxP = np.arange(st['p1']['positions'].shape[0])
        
    idx = np.intersect1d(idxC,idxP)
    
    return idx #1D array
            
        
        
    
    
        