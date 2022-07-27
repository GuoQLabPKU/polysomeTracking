import numpy as np 
import matplotlib.pyplot as plt
import collections 
import copy  

from nemotoc.py_io.tom_starread import tom_starread

def tom_plot_vectRiboClass(allTransList, if_norm = 1, if_show = 1, 
                           classList = None, save_dir = ''):
    '''
    TOM_PLOT_VECTRIBOCLASS plot the ribosome class of transformations
    '''
    if isinstance(allTransList, str):
        allTransList = tom_starread(allTransList)
        allTransList = allTransList['data_particles']
        
        
    stat_dict  = { }   
    #give a summary of the #particles of each particle class
    partClass = np.concatenate((allTransList['pairClass1'].values, allTransList['pairClass2'].values))
    partClassU = np.unique(partClass)
    #make a array to store the results of pairClass stat
    plotInfo_array = np.zeros((len(partClassU)*len(partClassU), 5))
    count_row = 0
    for i in partClassU:
        for j in partClassU:           
            plotInfo_array[count_row, 0] = i
            plotInfo_array[count_row, 1] = j           
            count_row += 1

    if classList is None:
        classList = np.unique(allTransList['pairClass'].values)

    for singleC in classList:
        transSingle = allTransList[allTransList['pairClass'] == singleC]
        #give a summary of the #particles of each particle class
        partClass = np.concatenate((transSingle['pairClass1'].values, transSingle['pairClass2'].values))
        partClassCount = collections.Counter(partClass)        
        for cl1, cl2 in zip(transSingle['pairClass1'].values, transSingle['pairClass2'].values):
            cl1Pos = np.where(plotInfo_array[:,0] == cl1)[0]
            cl2Pos = np.where(plotInfo_array[:,1] == cl2)[0]
            uPos = np.intersect1d(cl1Pos, cl2Pos)
            if len(uPos) > 0:
                plotInfo_array[uPos, 2] += 1
                plotInfo_array[uPos, 3] = partClassCount[cl1]
                plotInfo_array[uPos, 4] = partClassCount[cl2]
        
        stat_dict[singleC] = copy.deepcopy(plotInfo_array)
        #get the normalzied count and plotthe results 
        
        ax = plt.figure().gca()
        for singleRow in range(plotInfo_array.shape[0]):
            if if_norm:
                count_nor = plotInfo_array[singleRow, 2]/(plotInfo_array[singleRow, 3] + plotInfo_array[singleRow, 4])
                ax.plot([0, 1], [plotInfo_array[singleRow, 0], plotInfo_array[singleRow, 1]], 
                        alpha = count_nor*5, linewidth = count_nor*50, color = 'black')
            else:
                count_nor = plotInfo_array[singleRow, 2]
                ax.plot([0, 1], [plotInfo_array[singleRow, 0], plotInfo_array[singleRow, 1]], 
                        alpha = count_nor/20, linewidth = count_nor/20, color = 'black')                
        #plt.yticks([1,2,3,4,5], ['s1','s2','s3','s4','s5'], fontsize = 18)
        plt.xticks([0,1],['p1', 'p2'], fontsize = 18)
        #plt.ylabel('Class%d of transformations'%singleC, fontsize = 15)
        #recover the plotInfo_array    
        plt.title('Transformation class %d'%singleC)
        plotInfo_array[:, 2] = 0
        plotInfo_array[:, 3] = 0
        plotInfo_array[:, 4] = 0
        
        if len(save_dir) > 0:
            plt.savefig('%s/pairClassVis_cl%d.png'%(save_dir,singleC), dpi = 300)
        if if_show:
            plt.show()
        else:
            plt.close()
        
        
    return stat_dict
        
        
      