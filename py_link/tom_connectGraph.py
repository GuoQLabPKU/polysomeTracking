import networkx as nx
import pandas as pd
import numpy as np

def tom_connectGraph(transList):
    '''
    tom_connectGraph will return subgraph of which nodes are directly/indirectly
    connected with each other. Beside find the node w/o out or in-edges
    
    PATAMETER
    transList  transList with transform information stored
    
    Output
    statPerPoly_forfillUp  (dataframe)store the information of each polysome with 
                                ribosomes at the end of each polysome & ribosomes 
                                at the begin of each polysome. For polysome filling up!
    
    '''   
    allClasses = transList['pairClass'].values
    allTomos = transList['pairTomoID'].values
    tomosU = np.unique(allTomos)
    classesU = np.unique(allClasses)
    statePolyAll_forFillUp = pd.DataFrame({ })
    
    for single_class in classesU:           
        if single_class == 0:  
            continue  
        idx1 = np.where(allClasses == single_class)[0]
        offset_PolyID = 0
        for single_tomo in tomosU:
            idx2 = np.where(allTomos == single_tomo)[0]
            idx = np.intersect1d(idx1, idx2)
            #track the polysomes in the same tomogram and the same transform class
            if len(idx) >= 1:
                statePoly_forFillup,offset_PolyID = ribo_fillUp(transList.iloc[idx,:], offset_PolyID)
                #merge the dataframe
                statePolyAll_forFillUp = pd.concat([statePolyAll_forFillUp, statePoly_forFillup], axis = 0)
            
                
    #reset the idx of the data 
    statePolyAll_forFillUp.reset_index(inplace = True, drop = True)
    return statePolyAll_forFillUp          
            

def ribo_fillUp(pairList, offset_PolyID):
    #init the variables to store the ribosomes index w/o following(out-edge)
    #or followed ribosomes(in-edge)
    tomoID_list = [ ]
    tomoName_list = [ ]
    classNr_list = [ ]
    riboIDX_list = [ ]
    ifWoOut_list = [ ]
    ifWoIn_list = [ ]
    polyLabel_list = [ ]
    total_node = [ ]
    
    #get the graph representing polysomes
    pairIdx = pairList.loc[:, ['pairIDX1', 'pairIDX2']]
    pairTomoID = pairList['pairTomoID'].values[0]
    pairTomoName = pairList['pairTomoName'].values[0]
    pairClass = pairList['pairClass'].values[0]
    #contruct the network 
    ribosomeGraphs = nx.from_pandas_edgelist(pairIdx, source = 'pairIDX1', target = 'pairIDX2',
                                             create_using = nx.DiGraph)
    polysomeGroups = nx.weakly_connected_components(ribosomeGraphs)
    #get each polysome with branch or not
    count = 0
    for single_poly in polysomeGroups: #single_poly is set
        offset_PolyID += 1
        count += 1
        
        riboIdx_woOutList = [node for node in single_poly if ribosomeGraphs.out_degree(node) == 0]
        riboIdx_woInList =  [node for node in single_poly if ribosomeGraphs.in_degree(node) == 0]
        len_woOut = len(riboIdx_woOutList) 
        len_woIn = len(riboIdx_woInList)            
        polyLabel_list.extend([offset_PolyID]*(len_woIn + len_woOut))
        total_node.extend([len(single_poly)]*(len_woIn + len_woOut))
                
        if (len_woOut > 0) & (len_woIn > 0):
            riboIDX_list.extend(riboIdx_woOutList)
            riboIDX_list.extend(riboIdx_woInList)
            ifWoOut_list.extend([1]*len_woOut + [0]*len_woIn)

        elif (len_woOut > 0) & (len_woIn == 0):
            riboIDX_list.extend(riboIdx_woOutList)
            ifWoOut_list.extend([1]*len_woOut)
           
        elif (len_woOut == 0) & (len_woIn > 0):
            riboIDX_list.extend(riboIdx_woInList)
            ifWoOut_list.extend([0]*len_woIn)

        else:
            continue
    print('detect %d polysome groups in class%d in %s'%(count, pairClass, pairTomoName))
    
    tomoID_list.extend([pairTomoID]*len(ifWoOut_list))
    tomoName_list.extend([pairTomoName]*len(ifWoOut_list))
    classNr_list.extend([pairClass]*len(ifWoOut_list))
    ifWoIn_list = [-1*i+1 for i in ifWoOut_list]
    #make a dataframe & star file for further analysis
    statePoly_forFillup = pd.DataFrame({'pairTomoName':tomoName_list,
                                        'pairTomoID':tomoID_list, 'pairClass':classNr_list,
                                        'pairIDX':[int(i) for i in riboIDX_list], 'ifWoOut':ifWoOut_list,
                                        'ifWoIn':ifWoIn_list, 'pairLabel':polyLabel_list,
                                        'polylen_riboNr':total_node})
    
    return statePoly_forFillup,offset_PolyID