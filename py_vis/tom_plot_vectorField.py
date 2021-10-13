import numpy as np 
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from py_io.tom_extractData import tom_extractData
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_transform.tom_pointrotate import tom_pointrotate

def tom_plot_vectorField(posAng, mode = 'basic', tomoID = np.array([-1]), classNr = np.array([-1]), \
                         polyNr = np.array([-1]), onlySelected = 1, scale=20, \
                         repVect = np.array([[1,0,0]]), col = np.array([[0.7,0.7,0.7]]), outputFolder='',
                         if_2views = 0):
    
    type_ang = type(posAng)
    if (type_ang.__name__ == 'ndarray') | (type_ang.__name__ == 'str') | (type_ang.__name__ == 'DataFrame'):
        plot_vectField(posAng, mode, repVect, scale, col,  classNr, polyNr, onlySelected, tomoID, outputFolder,if_2views,type_ang)
    if isinstance(posAng, list):
        for posAngAct in posAng:
            plot_vectField(posAngAct, mode, repVect, scale, col, classNr, polyNr, onlySelected, tomoID, outputFolder,if_2views,type_ang)

def plot_vectField(posAng, mode, repVect, scale, col, classNr, polyNr, onlySelected, tomoID, outputFolder,if_2views,type_ang):
    if type_ang.__name__ == 'ndarray':
        pos = posAng[:,0:3]
        angles = posAng[:,3:6]
        ax = plt.figure().gca(projection ='3d')
        plotRepVects(pos, angles, repVect, scale, col, ax)
        plt.show()
        #plt.close()
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
        if (len(uTomoID) > 5) & (tomoID[0] == -1) & (len(outputFolder) == 0):  ##TomoID  = -1 ==> All tomo and only for vislization
            print('warning found %d tomograms reducing to 5'%len(uTomoID))
            print('can use tomoID parameter to select specific tomograms')
            uTomoID = uTomoID[:5]            
        
        print('rendering vector fields')
        
        for i in range(len(uTomoID)):
            tmpInd = np.where(allTomoID == uTomoID[i])[0]
            tomoName = allTomoLabel[tmpInd[0]]
            doRender(st, mode, classNr, polyNr, uTomoID, outputFolder,i,onlySelected, fTitleList,
                     tomoName, repVect, scale, col, if_2views)  #one tomo and one tomo,and plot each class in one tomo

def doRender(st, mode, classNr, polyNr, uTomoID, outputFolder,i,onlySelected, fTitleList, 
             tomoName, repVect, scale, col, if_2views):
    
    fTitle = '%s%s'%(fTitleList, tomoName)
    if len(outputFolder) > 0:
        if os.path.isdir(outputFolder):
            if not os.path.exists(outputFolder):
                os.mkdir(outputFolder)
        else:
            (shotdir,_) = os.path.split(outputFolder)
            if not os.path.exists(shotdir):
                os.mkdir(shotdir)
                   
        if if_2views:
            fig = plt.figure(figsize = (15,10))
            ax1 = fig.add_subplot(121, projection ='3d')
            ax2 = fig.add_subplot(122, projection ='3d')
        else:
            fig = plt.figure(figsize = (10,10))
            ax1 = fig.add_subplot(111, projection ='3d')
        
    else:
        if i >= 0:
            if_2views = 0
            fig = plt.figure(figsize = (10,10))
            ax1 = fig.add_subplot(111, projection ='3d')
            
    idx = filterList(st, classNr, polyNr, uTomoID[i])
    if len(idx) == 0:
        plt.close()
        return 
    if onlySelected:
        idxRep = idx
    else:
        idxRep = np.arange(0, len(st['p1']['positions'])) #plot the whole ribosomes
    
    if mode == 'advance':    
        pos = st['p1']['positions'][idxRep,:]
        angles = st['p1']['angles'][idxRep,:]
        plotRepVects(pos,angles, repVect, scale, col, ax1)
        if if_2views:
            plotRepVects(pos,angles, repVect, scale, col, ax2)
        
    if 'p2' in st.keys():  #plot another ribosome that near close (<2 ribosomo distance)
        pos = st['p2']['positions'][idxRep,:]
        angles = st['p2']['positions'][idxRep,:]
        if mode == 'advance':
            plotRepVects(pos, angles, repVect, scale, col, ax1)
            if if_2views:
                plotRepVects(pos,angles, repVect, scale, col, ax2)
        if if_2views:
            plotPairs(st, mode, idx, ax2)
        plotPairs(st, mode, idx, ax1)
        
        
    ax1.azim = 0
    ax1.elev = 90
    
    if if_2views:
        ax2.azim = 0
        ax2.elev = 0  
    if len(outputFolder)>0:
        fnameTmp = os.path.splitext(os.path.split(tomoName)[1])[0]
        fig.suptitle(fTitle)
        if os.path.isdir(outputFolder):
            plt.savefig('%s/%s.png'%(outputFolder,fnameTmp), dpi = 300)
        else:
            plt.savefig('%s'%outputFolder, dpi = 300)
#        plt.show()
        plt.close()
    else:
        fig.suptitle(fTitle)
#        plt.show()
        plt.close()

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
        if polyNr[0] == -1: #no poly is selected 
            idxP = np.arange(len(st['label']['pairClass']))
        else:
            allPoly = st['label']['pairLabel']
            idxP = np.where(np.fix(allPoly) == polyNr[:,None])[-1]  
    else:
        idxP = np.arange(st['p1']['positions'].shape[0])
    
    if 'tomoID' in st['label'].keys():
        if tomoID == -1: #no tomo is selected
            idxT = np.arange(len(st['label']['pairClass']))
        else:
            allTomo = st['label']['tomoID']
            idxT = np.where(allTomo == tomoID)[0]
    idx = np.intersect1d(idxP, idxC)
    idx = np.intersect1d(idx, idxT)
    return idx
            
        
def plotPairs(st, mode, idx, ax):
    allClasses = st['label']['pairClass'][idx]
    uClasses = np.unique(allClasses)
    allTrans = st['label']['p1p2TransVect'][idx,:]
    allCol = st['label']['pairClassColour'][idx,:]
    allPos1 = st['p1']['positions'][idx,:]
    allPos2 = st['p2']['positions'][idx,:]
    
    allLabel = st['label']['pairLabel'][idx]
    allPosInpoly1 = st['p1']['pairPosInPoly'][idx]
    allPosInpoly2 = st['p2']['pairPosInPoly'][idx]
    allClassP1 = st['p1']['classes'][idx]
    allClassP2 = st['p2']['classes'][idx]
    
    #make variables for legends
    h_plot = [ ]
    h_label = [ ]
    if_legend = 0
    for i in range(len(uClasses)):  #plot each trans class
        if uClasses[i] == 0:
            continue#I don't want to show class 0 --> noise!
        
        tmpIdx = np.where(allClasses == uClasses[i])[0]  #find the specific class of trans
        connVect = allTrans[tmpIdx,:]
        conCol = allCol[tmpIdx[0],:]
        conPos = allPos1[tmpIdx,:]
        midPos = conPos + connVect*0.35
        p2Pos = allPos2[tmpIdx,:]
        posInPoly1 = allPosInpoly1[tmpIdx]
        posInPoly2 = allPosInpoly2[tmpIdx]
        classInPoly1 = allClassP1[tmpIdx]
        classInPoly2 = allClassP2[tmpIdx]
        
        ax.quiver(conPos[:,0], conPos[:,1], conPos[:,2],
                  connVect[:,0], connVect[:,1], connVect[:,2], color = conCol,
                  arrow_length_ratio=0.4, linewidths = 0.7) #from each ribosome to the adaject pair
 
        if mode == 'advance':
            labelPoly = [ ]
            labelPosInPoly1 = [ ]
            labelPosInPoly2 = [ ]
            for ii in range(len(tmpIdx)):
                labelPoly.append('cl%s,p%.1f'%(uClasses[i],  allLabel[tmpIdx[ii]]))
                labelPosInPoly1.append('%d/c%d'%(posInPoly1[ii], classInPoly1[ii]))
                labelPosInPoly2.append('%d/c%d'%(posInPoly2[ii], classInPoly2[ii]))
           
            for x,y,z, lbl in zip(midPos[:,0], midPos[:, 1], midPos[:,2], labelPoly):
                ax.text(x,y,z,lbl, size = 10)
            for x,y,z, lbl in zip(conPos[:,0], conPos[:,1],conPos[:,2],labelPosInPoly1):
                ax.text(x,y,z,lbl, size = 10 )
            for x,y,z,lbl in zip(p2Pos[:,0],p2Pos[:,1],p2Pos[:,2], labelPosInPoly2):
                ax.text(x,y,z, lbl, size = 10) 
        
        if mode == 'basic':
            h_plot.append(ax.plot(conPos[0,0], conPos[0,1], conPos[0,2],color = conCol))
            h_label.append('cl%d'%(uClasses[i]))  
            
                       
        #plot for fillUp ribos
        allColClass = allCol[tmpIdx,:]
        fillIdx = np.where((allColClass == np.array([100,100,100])).all(1))[0]
        if len(fillIdx) == 0:
            if mode == 'advance':
                del labelPoly, labelPosInPoly1, labelPosInPoly2              
            continue   
        
        if_legend += 1
        conPosfill = conPos[fillIdx,:]
        connVectfill = connVect[fillIdx,:]
        ax.quiver(conPosfill[:,0], conPosfill[:,1], conPosfill[:,2],
                  connVectfill[:,0], connVectfill[:,1], connVectfill[:,2], linewidths = 1.5,
                  color = np.array([1,0,0]), arrow_length_ratio=0.4)
              
        if mode == 'advance':
            #change the color of label 
            for sIdx in fillIdx:
                ax.text(midPos[sIdx,0], midPos[sIdx,1], midPos[sIdx,2],
                        labelPoly[sIdx], size = 15, color = 'red')          
            del labelPoly, labelPosInPoly1, labelPosInPoly2
        
    #add filled up transform
    if (if_legend > 0) & (mode == 'basic'):
        h_plot.append(ax.plot(conPosfill[0,0], conPosfill[0,1], conPosfill[0,2], 
                               color = np.array([1,0,0])))
        h_label.append('fillup')   
    if  mode == 'basic':  
        ax.legend(h_plot,labels = h_label,fontsize = 10,#bbox_to_anchor=(1.15, 1),
                   title = 'class')  

def plotRepVects(pos,angles, repVect, scale, col,ax):
    for i in range(repVect.shape[0]):

        vectTmp = repVect[i,:]
        vectsRot = tom_rotVectByAng(vectTmp, angles)
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], vectsRot[:,0]*scale, \
                  vectsRot[:,1]*scale, vectsRot[:,2]*scale, length=1, color = col)
        
    
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
        