import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter

def tom_connectGraph(transList):
    '''
    tom_connectGraph will return subgraph of which nodes are directly/indirectly
    connected with each other. Beside find the node w/o out or in-edges.
    This linking strategy is transformation class insenstive(no need consider class)
    
    PATAMETER
    transList  transList with transform information stored
    
    Output
    statPerPoly  (dataframe)store the information of each polysome with 
                 ribosomes at the end of each polysome & ribosomes 
                 at the begin of each polysome. 
    
    '''   
    allTomos = transList['pairTomoID'].values
    tomosU = np.unique(allTomos)
    pairLabel_dataFrame = pd.DataFrame({})
    statePoly_dict = { }
    offset_PolyID = 0
    for single_tomo in tomosU:
        idx = np.where(allTomos == single_tomo)[0]
        #track the polysomes in the same tomogram 
        if len(idx) >= 1:
            pairLabelSingleTomo_dataFrame, offset_PolyID, statePoly_dict = nx_polysomeGroup(transList.iloc[idx,:], offset_PolyID, statePoly_dict)
            #merge the dataframe
            pairLabel_dataFrame = pd.concat([pairLabel_dataFrame, pairLabelSingleTomo_dataFrame], axis = 0)
            
                
    #merge these twoo dataframes
    pairLabel_dataFrame.columns = ['pairIDX1', 'pairIDX2', 'pairLabely']
    assert -1 not in pairLabel_dataFrame['pairLabely']
    transList_update = pd.merge(transList, pairLabel_dataFrame, left_on = ['pairIDX1','pairIDX2'],
                                right_on = ['pairIDX1','pairIDX2'])
    transList_update['pairLabel'] = transList_update['pairLabely']
    transList_update.drop(['pairLabely'], axis = 1, inplace = True)
    transList_update.reset_index(inplace = True, drop = True)
    #also write the summary of polysome 
    statePoly_dataFrame = pd.DataFrame(statePoly_dict)
    statePoly_dataFrame['name'] = ['endRiboNr', 'beginRiboNr', 'outBranchNr', 'inBranchNr','riboNr', 'mixClassNr', 'if_dimerIn']
    statePoly_dataFrame.set_index('name',inplace = True)
    statePoly_dataFrame_sort = statePoly_dataFrame.sort_values(by = ['riboNr'], axis = 1, ascending = False)
    return transList_update, statePoly_dataFrame_sort    
            

def nx_polysomeGroup(pairList, offset_PolyID, statePoly_dict):
    #init the variables to store the ribosomes index w/o following(out-edge)
    #or followed ribosomes(in-edge)
    
    #get the graph representing polysomes
    pairIdx = pairList.loc[:, ['pairIDX1', 'pairIDX2', 'pairClass']]
    pairLabelSingleTomo_dataFrame = pairList.loc[:, ['pairIDX1', 'pairIDX2', 'pairLabel']]
    pairTomoName = pairList['pairTomoName'].values[0]
    #contruct the network 
    ribosomeGraphs = nx.from_pandas_edgelist(pairIdx, source = 'pairIDX1', target = 'pairIDX2',
                                             create_using = nx.DiGraph)
    polysomeGroups = nx.weakly_connected_components(ribosomeGraphs)
    #get each polysome with branch or not
    count = 0
    for single_poly in polysomeGroups: #single_poly is set   
        poly_transClassList = [ ]
        offset_PolyID += 1
        count += 1
        #get the polysome summary information
        riboIdx_woOutList = [node for node in single_poly if ribosomeGraphs.out_degree(node) == 0]
        riboIdx_woInList =  [node for node in single_poly if ribosomeGraphs.in_degree(node) == 0]
        
        riboIdx_branchOutList = [node for node in single_poly if ribosomeGraphs.out_degree(node) >= 2]
        riboIdx_branchInList =  [node for node in single_poly if ribosomeGraphs.in_degree(node) >= 2]
        len_woOut = len(riboIdx_woOutList) 
        len_woIn = len(riboIdx_woInList)
        len_branchOut = len(riboIdx_branchOutList) 
        len_branchIn = len(riboIdx_branchInList)
        
        #get the dataframe to store the information of pairLabel for each transformation
        for singleNode in single_poly:
            nextNodes = list(ribosomeGraphs.successors(singleNode))
            for singleNext in nextNodes:
                whereSingleNode = np.where(pairLabelSingleTomo_dataFrame['pairIDX1'] == singleNode)[0]
                whereNextNode = np.where(pairLabelSingleTomo_dataFrame['pairIDX2'] == singleNext)[0]
                wherePair = np.intersect1d(whereSingleNode, whereNextNode)
                if len(wherePair)>0:
                    pairLabelSingleTomo_dataFrame.iloc[wherePair[0],2] = offset_PolyID
                    poly_transClassList.append(pairIdx.iloc[wherePair[0],2])
                    
        poly_transClassList_count = dict(Counter(poly_transClassList))
        if (6 in poly_transClassList_count.keys()) | (11 in poly_transClassList_count.keys()):
            singlePolyInfo_list = [len_woOut,len_woIn,len_branchOut, len_branchIn, len(single_poly), poly_transClassList_count.keys().__len__(),1]            
        else:
            singlePolyInfo_list = [len_woOut,len_woIn,len_branchOut, len_branchIn, len(single_poly), poly_transClassList_count.keys().__len__(),0]                        
        statePoly_dict[offset_PolyID] = singlePolyInfo_list
    #return the data 
    #print('detect %d polysomes in %s'%(count, pairTomoName))
    return pairLabelSingleTomo_dataFrame, offset_PolyID, statePoly_dict
    
