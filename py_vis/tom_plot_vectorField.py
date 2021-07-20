import numpy as np 
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from py_io.tom_extractData import tom_extractData
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_transform.tom_pointrotate import tom_pointrotate

def tom_plot_vectorField(posAng, tomoID = np.array([-1]), classNr = np.array([-1]), \
                         polyNr = np.array([-1]), onlySelected = 1, scale=20, \
                         repVect = np.array([[1,0,0]]), col = np.array([0,0,1]), cmbInd = '', outputFolder=''):
    
    type_ang = type(posAng)
    if (type_ang.__name__ == 'ndarry') | (type_ang.__name__ == 'str') | (type_ang.__name__ == 'DataFrame'):
        plot_vectField(posAng, repVect, scale, col, cmbInd, classNr, polyNr, onlySelected, tomoID, outputFolder, type_ang)
    if type_ang == 'list':
        for posAngAct in posAng:
            plot_vectField(posAngAct, repVect, scale, col, cmbInd, classNr, polyNr, onlySelected, tomoID, outputFolder,type_ang)

def plot_vectField(posAng, repVect, scale, col, cmbInd, classNr, polyNr, onlySelected, tomoID, outputFolder, type_ang):
    if type_ang.__name__ == 'ndarry':
        fTitle = ''
        pos = posAng[:,0:3]
        angles = posAng[:,3:6]
        plotRepVects(pos,angles,repVect, scale, col, fTitle)
        return 
    if (type_ang.__name__ == 'str') | (type_ang.__name__ == 'DataFrame'):
        fTitleList = ''
        
        if type_ang.__name__ == 'str':
            fTitleList = '%s '%posAng
        st = tom_extractData(posAng)
        allTomoID = st['label']['tomoID']
        uTomoID = np.unique(allTomoID)
        allTomoLabel = st['label']['tomoName']
        
        
        if tomoID[0] > -1:  #select this tomo
            uTomoID = tomoID  #only keep this tomoID further analysis
        if len(uTomoID) > 1:
            print('more than one tomogram in list')          
            for i in range(len(uTomoID)):
                tmpInd = np.where(allTomoID == uTomoID[i])[0]
                print('tomoID: %d  tomoName: %s'%(uTomoID[i], allTomoLabel[tmpInd[0]]))
        if (len(uTomoID) > 5) & (tomoID[0] == -1) & (len(outputFolder) ==0):  ##TomoID  = -1 ==> All tomo
            print('warning found %d tomograms reducing to 5'%len(uTomoID))
            print('use tomoID parameter to select specific tomograms')
            uTomoID = uTomoID[:5]
            
        plotClassZero = len(np.where(classNr == 0)[0]) > 0
        print('rendering vector fields')
        
        for i in range(len(uTomoID)):
            tmpInd = np.where(allTomoID == uTomoID[i])[0]
            tomoName = allTomoLabel[tmpInd[0]]
            doRender(st, classNr, polyNr, uTomoID, outputFolder,i,onlySelected, fTitleList,
                     tomoName, repVect, scale, col,plotClassZero )  #one tomo and one tomo,and plot each class in one tomo

def doRender(st, classNr, polyNr, uTomoID, outputFolder,i,onlySelected, fTitleList, 
             tomoName, repVect, scale, col, plotClassZero):
    
    fTitle = '%s%s'%(fTitleList, tomoName)
    if len(outputFolder) > 0:
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        fig = plt.figure()
        ax = fig.gca(projection ='3d')
        
    else:
        if i >= 0:
            ax = plt.figure().gca(projection ='3d')
            
    idx = filterList(st, classNr, polyNr, uTomoID[i])
    if len(idx) == 0:
        plt.close()
        return 
    if onlySelected:
        idxRep = idx
    else:
        idxRep = np.arange(0, len(st['p1']['positions'])) #plot the whole ribosomes
        
    pos = st['p1']['positions'][idxRep,:]
    angles = st['p1']['angles'][idxRep,:]
    plotRepVects(pos,angles, repVect, scale, col, ax)
        
    if 'p2' in st.keys():  #plot another ribosome that near close (<2 ribosomo distance)
        pos = st['p2']['positions'][idxRep,:]
        angles = st['p2']['positions'][idxRep,:]
        plotRepVects(pos, angles, repVect, scale, col, ax)
        plotPairs(st, idx, plotClassZero, ax)
        
    
    
    if len(outputFolder)>0:
        fnameTmp = os.path.splitext(os.path.split(tomoName)[1])[0]
        plt.title(fTitle)
        plt.savefig('%s/%s.png'%(outputFolder,fnameTmp), dpi = 300)
        plt.show()
        #plt.close()
    else:
        plt.title(fTitle)
        plt.show()
        #plt.close()

def filterList(st, classNr, polyNr, tomoID):
    if 'pairClass' in st['label'].keys():
        if classNr[0] == -1:  ##no any clustering perform
            idxC = np.arange(len(st['label']['pairClass']))
        else:
            allClasses = st['label']['pairClass']
            idxC = np.where(allClasses==classNr[:,None])[-1]
    else:
        idxC = np.arange(st['p1']['positions'].shape[0])
    
    if 'pairLabel' in st['label'].keys():
        if polyNr == -1:
            idxP = np.arange(len(st['label']['pairClass']))
        else:
            allPoly = st['label']['pairLabel']
            idxP = np.where(np.fix(allPoly) == polyNr[:,None])[-1]  
    else:
        idxP = np.arange(st['p1']['positions'].shape[0])
    
    if 'tomoID' in st['label'].keys():
        if tomoID == -1:
            idxT = np.arange(len(st['label']['pairClass']))
        else:
            allTomo = st['label']['tomoID']
            idxT = np.where(allTomo == tomoID)[0]
    idx = np.intersect1d(idxP, idxC)
    idx = np.intersect1d(idx, idxT)
    return idx
            
        
def plotPairs(st, idx, plotClassZero, ax):
    allClasses = st['label']['pairClass'][idx]
    uClasses = np.unique(allClasses)
    allTrans = st['label']['p1p2TransVect'][idx,:]
    allCol = st['label']['pairClassColour'][idx,:]
    allPos = st['p1']['positions'][idx,:]
    allPos2 = st['p2']['positions'][idx,:]
    
    allLabel = st['label']['pairLabel'][idx]
    allPosInpoly1 = st['p1']['pairPosInPoly'][idx]
    allPosInpoly2 = st['p2']['pairPosInPoly'][idx]
    allClassP1 = st['p1']['classes'][idx]
    allClassP2 = st['p2']['classes'][idx]
    
    for i in range(len(uClasses)):  #plot each trans class
        if uClasses[i] == 0:
            continue#I don't want to show class 0 --> noise!
            if plotClassZero == 0:
                continue
        
        tmpIdx = np.where(allClasses == uClasses[i])[0]  #find the specific class of trans
        connVect = allTrans[tmpIdx,:]
        conCol = allCol[tmpIdx[0],:]
        conPos = allPos[tmpIdx,:]
        midPos = conPos + connVect*0.35
        p2Pos = allPos2[tmpIdx,:]
        posInPoly1 = allPosInpoly1[tmpIdx]
        posInPoly2 = allPosInpoly2[tmpIdx]
        classInPoly1 = allClassP1[tmpIdx]
        classInPoly2 = allClassP2[tmpIdx]
        
        ax.quiver(conPos[:,0], conPos[:,1], conPos[:,2],
                  connVect[:,0], connVect[:,1], connVect[:,2], color = conCol) #from each ribosome to the adaject pair
        
        labelPoly = np.repeat('-1',len(tmpIdx))
        labelPosInPoly1 = np.repeat('-1',len(tmpIdx)) 
        labelPosInPoly2 = np.repeat('-1',len(tmpIdx))
        for ii in range(len(tmpIdx)):
            labelPoly[ii] = 'c%d,p%d'%(uClasses[i],  int(allLabel[tmpIdx[ii]]))
            labelPosInPoly1[ii] = '%d/%d'%(posInPoly1[ii], classInPoly1[ii])
            labelPosInPoly2[ii] = '%d/%d'%(posInPoly2[ii], classInPoly2[ii])
        
        for x,y,z, lbl in zip(midPos[:,0], midPos[:, 1], midPos[:,2], labelPoly):
            ax.text(x,y,z,lbl, size = 10)
        for x,y,z, lbl in zip(conPos[:,0], conPos[:,1],conPos[:,2],
                labelPosInPoly1):
            ax.text(x,y,z,lbl, size = 10 )
        for x,y,z,lbl in zip(p2Pos[:,0],p2Pos[:,1],p2Pos[:,2], 
                labelPosInPoly2):
            ax.text(x,y,z, lbl, size = 10)
        
        
        #plot for fillUp ribos
        allColClass = allCol[tmpIdx,:]
        fillIdx = np.where((allColClass == np.array([0,0,1])).all(1))[0]
        if len(fillIdx) == 0:
            del labelPoly, labelPosInPoly1, labelPosInPoly2
            continue
        conPosfill = conPos[fillIdx,:]
        connVectfill = connVect[fillIdx,:]
        ax.quiver(conPosfill[:,0], conPosfill[:,1], conPosfill[:,2],
                  connVectfill[:,0], connVectfill[:,1], connVectfill[:,2], linewidths = 8,
                  color = np.array([1.0,0.0,0.0]))
        #change the color of label 
        for sIdx in fillIdx:
            ax.text(midPos[sIdx,0],midPos[sIdx,1],midPos[sIdx,2],labelPoly[sIdx], size = 15,color = 'red')
           

        del labelPoly, labelPosInPoly1, labelPosInPoly2
        
        
        
        
        
def plotRepVects(pos,angles, repVect, scale, col,ax):
    for i in range(repVect.shape[0]):

        vectTmp = repVect[i,:]
        vectsRot = tom_rotVectByAng(vectTmp, angles)
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], vectsRot[:,0]*scale, \
                  vectsRot[:,1]*scale, vectsRot[:,2]*scale,length=1,color = col)
        
        

def tom_rotVectByAng(vect, angs, rotFlav = 'zxz', display = None, col = np.array([0,0,1])):
    vectsRot = np.zeros((angs.shape[0], 3))
    for i in range(angs.shape[0]):
        if rotFlav == 'zyz':
            angTmp = tom_eulerconvert_xmipp(angs[i,0], angs[i,1], angs[i,2])
        else:
            angTmp = angs[i,:]
        vectsRot[i,:] = tom_pointrotate(vect, angTmp[0], angTmp[1], angTmp[2])
    
    if display == 'vector':
        ori = np.zeros( (vectsRot.shape[0], vectsRot.shape[0]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.quiver(ori[:,0], ori[:,1], ori[:,2], 
                  vectsRot[:,0], vectsRot[:,1], vectsRot[:,2],
                  color = col)
        plt.show()
        plt.close()
    if display == 'points':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.plot(vectsRot[:,0], vectsRot[:,1], vectsRot[:,2],
                'ro',color = col)
        plt.show()
        plt.close()
    return vectsRot
        