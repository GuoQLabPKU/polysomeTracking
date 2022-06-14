import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial

from py_io.tom_starread import tom_starread,generateStarInfos
from py_io.tom_starwrite import tom_starwrite
from py_transform.tom_pointrotate import tom_pointrotate
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp

def tom_plot_mRNAPath(allTransList, relCoordA = np.array([15,1,3]), relCoordE = np.array([6,-13,7]), 
             saveFileName = '', figsave_dir = '', classList = None, minPolyLen = 10, if_disp = 1):
    '''
    TOM_MRNAPATH plot the mRNA path given a polysome transList,
    the relative cooridnates of A-E site of the center point are need
    '''
    if isinstance(allTransList, str):
        allTransList = tom_starread(allTransList)
        allTransList = allTransList['data_particles']
    absCoordA1X = [ ]
    absCoordA1Y = [ ]
    absCoordA1Z = [ ]
    absCoordA2X = [ ]
    absCoordA2Y = [ ]
    absCoordA2Z = [ ]
    absCoordE1X = [ ]
    absCoordE1Y = [ ]
    absCoordE1Z = [ ]
    absCoordE2X = [ ]
    absCoordE2Y = [ ]
    absCoordE2Z = [ ]
    pairIdx1 = [ ]
    pairIdx2 = [ ]
    polyClassList = [ ]
    polyLabelList = [ ]
    linkType = [ ]
    if classList == None:
        classList = np.unique(allTransList['pairClass'].values)
    for singleCl in classList:
        singleClTrans = allTransList[allTransList['pairClass'] == singleCl]
        polyLen = singleClTrans['pairLabel'].value_counts()
        polyLabel = polyLen[polyLen >= minPolyLen].index
        for singlePoly in polyLabel:
            singlePolyTrans = singleClTrans[singleClTrans['pairLabel'] == singlePoly]
            for singleRow in range(singlePolyTrans.shape[0]):
                singlePair = singlePolyTrans.iloc[singleRow, :]
                pos1 = singlePair[['pairCoordinateX1', 'pairCoordinateY1', 'pairCoordinateZ1']].values
                ang1 = singlePair[['pairAnglePhi1', 'pairAnglePsi1', 'pairAngleTheta1']].values
                pos2 = singlePair[['pairCoordinateX2', 'pairCoordinateY2', 'pairCoordinateZ2']].values
                ang2 = singlePair[['pairAnglePhi2', 'pairAnglePsi2', 'pairAngleTheta2']].values
                #get the absCoord
                absCoordA1, absCoordE1, absCoordA2, absCoordE2, linkForm = linkTransPair(pos1, ang1, 
                                                                                         pos2, ang2, 
                                                                                         relCoordA, 
                                                                                         relCoordE)
                absCoordA1X.append(absCoordA1[0]); absCoordA1Y.append(absCoordA1[1]); absCoordA1Z.append(absCoordA1[2])
                absCoordE1X.append(absCoordE1[0]); absCoordE1Y.append(absCoordE1[1]); absCoordE1Z.append(absCoordE1[2])
                absCoordA2X.append(absCoordA2[0]); absCoordA2Y.append(absCoordA2[1]); absCoordA2Z.append(absCoordA2[2])
                absCoordE2X.append(absCoordE2[0]); absCoordE2Y.append(absCoordE2[1]); absCoordE2Z.append(absCoordE2[2])
                pairIdx1.append(singlePair['pairIDX1'])
                pairIdx2.append(singlePair['pairIDX2'])
                polyClassList.append(singleCl)
                polyLabelList.append(singlePoly)
                linkType.append(linkForm)
    #store and make the figure 
    mRNAPathInfo = pd.DataFrame({'pairIdx1':pairIdx1, 'pairIdx2':pairIdx2, 
                                 'coordAX1':absCoordA1X, 'coordAY1':absCoordA1Y, 'coordAZ1':absCoordA1Z,
                                 'coordEX1':absCoordE1X, 'coordEY1':absCoordE1Y, 'coordEZ1':absCoordE1Z,                                                                                                  
                                 'coordAX2':absCoordA2X, 'coordAY2':absCoordA2Y, 'coordAZ2':absCoordA2Z,
                                 'coordEX2':absCoordE2X, 'coordEY2':absCoordE2Y, 'coordEZ2':absCoordE2Z,
                                 'polyClass':polyClassList, 'polyLabel':polyLabelList,'linkType':linkType})            
            
    if len(saveFileName) > 0:
        mRNA = generateStarInfos()
        mRNA['data_particles'] = mRNAPathInfo
        tom_starwrite(saveFileName, mRNA)

    if if_disp:
        #split into different single polysome          
        polyClass = np.unique(mRNAPathInfo['polyClass'].values)
        for singleC in polyClass:
            mRNAClass = mRNAPathInfo[mRNAPathInfo['polyClass'] == singleC]
            polyLabel = np.unique(mRNAClass['polyLabel'].values)
            for singleLabel in polyLabel:
                mRNApoly = mRNAClass[mRNAClass['polyLabel'] == singleLabel]
                fig = plt.figure(figsize = (10,10))
                ax = fig.add_subplot(111, projection ='3d')
                mRNApoly['coordEY1'] = 6000-mRNApoly['coordEY1'].values
                mRNApoly['coordEY2'] = 6000-mRNApoly['coordEY2'].values
                mRNApoly['coordAY1'] = 6000-mRNApoly['coordAY1'].values
                mRNApoly['coordAY2'] = 6000-mRNApoly['coordAY2'].values
                showmRNAPath(mRNApoly, ax)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                plt.axis('off')
                plt.legend(bbox_to_anchor=(1.00, 1),
                           fontsize=20,edgecolor='black')
                if len(figsave_dir) > 0:
                    plt.savefig('%s/cl%d_poly%d.png'%(figsave_dir, singleC, singleLabel), dpi = 300)
                plt.show()
    return mRNAPathInfo

def showmRNAPath(mRNAPath, ax=None, colorArrowinner = np.array([[0.7,0.7,0.7]]), colorArrowinter = np.array([0.7,0.7,0.7]),
                 arrow_length_ratio = 0.4, if_disp = 1, if_calcCos=1, if_changeLinkType=0):
    if ax is None:
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection ='3d') 
        ax.grid(False)
        plt.axis('off')
    #plot the arrow from E1 to A1
    E1X = mRNAPath['coordEX1'].values;E1Y = mRNAPath['coordEY1'].values;E1Z = mRNAPath['coordEZ1'].values
    E12A1X = mRNAPath['coordAX1'].values - mRNAPath['coordEX1'].values
    E12A1Y = mRNAPath['coordAY1'].values - mRNAPath['coordEY1'].values
    E12A1Z = mRNAPath['coordAZ1'].values - mRNAPath['coordEZ1'].values            
    ax.quiver(E1X, E1Y, E1Z,E12A1X, E12A1Y, E12A1Z, color = colorArrowinner,
              arrow_length_ratio=arrow_length_ratio, linewidths =2.5)
    
    E2X = mRNAPath['coordEX2'].values;E2Y = mRNAPath['coordEY2'].values;E2Z = mRNAPath['coordEZ2'].values
    E22A2X = mRNAPath['coordAX2'].values - mRNAPath['coordEX2'].values
    E22A2Y = mRNAPath['coordAY2'].values - mRNAPath['coordEY2'].values
    E22A2Z = mRNAPath['coordAZ2'].values - mRNAPath['coordEZ2'].values     
    ax.quiver(E2X, E2Y, E2Z, E22A2X, E22A2Y, E22A2Z, color = colorArrowinner,
              arrow_length_ratio=arrow_length_ratio, linewidths =2.5)
    
    #plot the connection between close AE site 
    linkType = mRNAPath['linkType'].values[0]
    if if_changeLinkType: ##this is designed for class6 di-ribsome
        if linkType =='A2E1':
            linkType = 'A1E2'
        else:
            linkType = 'A2E1'
    
    if linkType == 'A2E1':
        A2X = mRNAPath['coordAX2'].values;A2Y = mRNAPath['coordAY2'].values;A2Z = mRNAPath['coordAZ2'].values
        A22E1X = mRNAPath['coordEX1'].values - mRNAPath['coordAX2'].values
        A22E1Y = mRNAPath['coordEY1'].values - mRNAPath['coordAY2'].values
        A22E1Z = mRNAPath['coordEZ1'].values - mRNAPath['coordAZ2'].values       
        ax.quiver(A2X, A2Y, A2Z, A22E1X, A22E1Y, A22E1Z, color = colorArrowinter,
                  arrow_length_ratio=arrow_length_ratio, linewidths = 2.5)        
        
    if linkType == 'A1E2':
        A1X = mRNAPath['coordAX1'].values;A1Y = mRNAPath['coordAY1'].values;A1Z = mRNAPath['coordAZ1'].values
        A12E2X = mRNAPath['coordEX2'].values - mRNAPath['coordAX1'].values
        A12E2Y = mRNAPath['coordEY2'].values - mRNAPath['coordAY1'].values
        A12E2Z = mRNAPath['coordEZ2'].values - mRNAPath['coordAZ1'].values       
        ax.quiver(A1X, A1Y, A1Z, A12E2X, A12E2Y, A12E2Z, color = colorArrowinter,
                  arrow_length_ratio=arrow_length_ratio, linewidths = 2.5)
    
    #label A site and E site
    ax.scatter(E1X, E1Y, E1Z, color = 'firebrick', label = 'E sites', s = 60)
    ax.scatter(E2X, E2Y, E2Z, color = 'firebrick', s = 60)
    ax.scatter(mRNAPath['coordAX1'].values, 
               mRNAPath['coordAY1'].values, 
               mRNAPath['coordAZ1'].values, 
               color = 'royalblue', label = 'A sites', s = 60)
    ax.scatter(mRNAPath['coordAX2'].values, 
               mRNAPath['coordAY2'].values, 
               mRNAPath['coordAZ2'].values, 
               color = 'royalblue', s = 60)
#    ax.set_xlabel('x', fontsize = 15)
#    ax.set_ylabel('y', fontsize = 15)
#    ax.set_zlabel('z', fontsize = 15)

    if not if_disp:
        plt.close()
    #calculate the cos distance between this mRNA knots 
    if if_calcCos:
        if linkType == 'A2E1':
            cosSim11, cosSim21 = calCos(A22E1X, A22E1Y, A22E1Z,E22A2X, E22A2Y, E22A2Z)
            cosSim12, cosSim22 = calCos(E12A1X, E12A1Y, E12A1Z,A22E1X, A22E1Y, A22E1Z)
        else:
            cosSim11, cosSim21 = calCos(A12E2X, A12E2Y, A12E2Z,E12A1X, E12A1Y, E12A1Z)
            cosSim12, cosSim22 = calCos(E22A2X, E22A2Y, E22A2Z,A12E2X, A12E2Y, A12E2Z)
            
        return ax, np.array([cosSim11, cosSim12,cosSim21, cosSim22])
    else:
        return ax
   
def calCos(arrayX1, arrayY1, arrayZ1, arrayX2, arrayY2, arrayZ2):
    cosSimList = [ ]
    for x1,y1,z1,x2,y2,z2 in zip(arrayX1, arrayY1, arrayZ1, arrayX2, arrayY2, arrayZ2):
        cosSimList.append(1-spatial.distance.cosine([x1,y1,z1],[x2,y2,z2]))
    
    return cosSimList[0], cosSimList[1]
        
        
           
def linkTransPair(pos1,ang1,pos2,ang2, relCoordA, relCoordE):
    absCoordA1, absCoordE1 = returnAEsite(relCoordA, relCoordE, pos1, ang1, 'tom')
    absCoordA2, absCoordE2 = returnAEsite(relCoordA, relCoordE, pos2, ang2, 'tom')
    A1E2 = np.linalg.norm(absCoordA1 - absCoordE2)
    A2E1 = np.linalg.norm(absCoordA2 - absCoordE1)
    if A1E2 > A2E1:
        return absCoordA1, absCoordE1, absCoordA2, absCoordE2, 'A2E1'
    else:
        return absCoordA1, absCoordE1, absCoordA2, absCoordE2, 'A1E2'
    



def returnAEsite(relCoordA, relCoordE, absCoordCenter, rotateAngle, flag = 'xmipp'):
    if flag == 'xmipp':
        _, rotateAngle = tom_eulerconvert_xmipp(rotateAngle[0], rotateAngle[1], rotateAngle[2], 'xmipp2tom')
    
    absCoordA = tom_pointrotate(relCoordA, rotateAngle[0], rotateAngle[1], rotateAngle[2]) + absCoordCenter
    absCoordE = tom_pointrotate(relCoordE, rotateAngle[0], rotateAngle[1], rotateAngle[2]) + absCoordCenter
    
    return absCoordA, absCoordE
        
    


    