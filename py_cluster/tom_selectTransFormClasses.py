from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_extractData import tom_extractData
import numpy as np
import os

def tom_selectTransFormClasses(transList, selList, outputFolder = '', minNumTransForms = -1):
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
                selList[i]["classNr"] = uClass[i] #can be list of single int, the same for polyNr
                selList[i]["polyNr"] = -1 #selList is a list with dicts stored
    
    zz = 1
    idxCmb = [ ]
    transListSel = [ ]
    selFolder = [ ]
    for i in range(len(selList)):#process each class, also class0:fail to cluster
        clNr = selList[i]["classNr"]
        polyNr = selList[i]['polyNr']
        idx = filterList(st, clNr, polyNr) #which class and in this class,which polysome you like?
        if clNr == 0:
            continue #garbege class
        if len(idx) < minNumTransForms:
            continue #this class has few transforms 
        transListSel.append( transList.iloc[idx,:])
        trFilt = transList.iloc[idx,:]
        selFolder.append(genOutput(trFilt, selList[i], outputFolder))
        zz+=1
        idxCmb = np.concatenate((idxCmb, idx), axis = 0)
        
    transListSelCmb = transList.iloc[idxCmb,:]




def genOutput(transList, selList, outputFolder):
    if outputFolder == '':
        selFolder = '-1'
        return selFolder
    clNr = selList['classNr']
    polyNr = selList['polyNr']
    
    if polyNr[0] == -1:
        polyNr = ''
    else:
        if type(polyNr).__name__ == 'int':
            polyNrStr = str(polyNr)
        elif (type(polyNr).__name__ == 'list') | (type(polyNr).__name__ == 'ndarray'):
            polyNrStr = '+'.join([str(i) for i in polyNr]) #in any case, there  only less than one '+'
        polyNrStr = 'p%s'%polyNrStr
        
    if type(clNr).__name__ == 'int':
            clNrStr = str(clNr)
    elif (type(clNr).__name__ == 'list') | (type(clNr).__name__ == 'ndarray'):
            clNrStr = '+'.join([str(i) for i in clNr]) #in any case, there  only less than one '+'
    selFolder = '%s/c%s%s'%(outputFolder, clNrStr, polyNrStr)
    os.mkdir(selFolder)
    #write starfile
    tom_starwrite('%s/transList.star', transList, list(transList.columns))
    return selFolder
 
    


def filterList(st, classNr, polyNr):
    if 'pairClass' in st["label"].keys():
        if classNr == -1:
            idxC = np.arange(len(st['label']['pairClass']))
        else:
            allClasses = st["label"]["pairClass"]
            #judge the type of input:
            if type(classNr).__name__ == 'int':
                idxC = np.where(allClasses == classNr) #return the index of classNr
            elif (type(classNr).__name__ == 'ndarray') |  (type(classNr).__name__ == 'list'):
                idxC = np.where(allClasses==classNr[:,None])[-1] #in case the classNr is an container
                
    else:
        idxC = np.arange(st['p1']['positions'].shape[0])
    if 'pairLabel' in st["label"].keys():
        if polyNr == -1:
            idxP = np.arange(len(st['label']['pairClass']))
        else:               
            allPoly = st["label"]["pairlabel"]
            if type(polyNr).__name__ == 'int':
                idxP = np.where(allPoly == polyNr) #return the index of classNr
            elif (type(polyNr).__name__ == 'ndarray') |  (type(classNr).__name__ == 'list'):
                idxP = np.where(allPoly == polyNr[:,None])[-1] #in case the classNr is an container            
   
    else:
        idxP = np.arange(st['p1']['positions'].shape[0])
    idx = np.intersect1d(idxC,idxP)
    
    return idx #1D array
            
        
        
    
    
        