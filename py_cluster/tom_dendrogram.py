from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns


def tom_dendrogram(tree,ColorThreshold = -1, nrObservation = -1,dsp = 1,maxLeaves = 0):
    '''
    TOM_DENDROGRAM creates a dendrogram from linked data and gives the members
    and color for clusters

    [groups,cmap,groupIdx,ColorThreshold,lineHandle]=tom_dendrogram(tree,ColorThreshold,nrObservations,dsp)

    PARAMETERS

    INPUT
       tree                          linked data returned by the tom_calcLinkage()
       ColorThreshold                (mean(tree(:,3))) threshold for creating clusters
                                      can be number/auto/off(don't cluster)
       nrObservations                (-1) number of obs b4 linkage needed for cluster  #shoule be precision
                                          members. It should == #of ribosomes pairs
                                          (rem:more than two members can be merged at the same time)
       dsp                           (1) display flag
       maxLeaves                     (7000) max number of leaves in dendrogram to display use 0
                                                      to switch off
   

   OUTPUT
       groups           		     list containing members,color and id for
                                          each cluster
       cmap                          used colors
       groupIdx                      index of groups
       ColorThreshold                threshold applyed
       hline                         handle to lines     

   EXAMPLE



   REFERENCES

   SEE ALSO
       linkage,dendrogram
   '''
    if isinstance(tree,str):
        tree = np.load(tree)
        
    if ColorThreshold == -1:
        ColorThreshold = np.max(tree[:,2])*0.7 #default threshold for hierachical clustering
    if ColorThreshold == 'auto':
        del ColorThreshold
        ColorThreshold = np.max(tree[:,2])*0.7
    if nrObservation == -1:
        nrObservation = tree.shape[0] + 1

    tree_cp = deepcopy(tree)  #this is necessary, I will change the tree following 
    groupIdx, _, cmap = genColorLookUp(tree_cp, ColorThreshold) #the tree changed!
 
    
    
    if len(groupIdx) == 0:
        groups = ''
        return groups, cmap, groupIdx, ColorThreshold  #if the threshold is too high/0,
    
    groups = []                                                       #no classes can be checked
    dlabels = np.array(['c0']*nrObservation)   #it should be nrObservation = size(cluster_tree) + 1
    ugroupIdx = np.unique(groupIdx) #for example we assume that genColorLookUp returns
    len_ugroupidx = len(ugroupIdx)
    
    for i in range(len_ugroupidx):
        groups.append({})
        groups[i]["id"] = ugroupIdx[i] #belongs to which cluster
        if ugroupIdx[i] == 0:
            groups[i]["color"] = cmap[-1,:] #0 means that it belongs to no cluster(impossible)
        else:
            groups[i]["color"] = cmap[ugroupIdx[i]-1, :]
        
        tt = np.where(groupIdx == ugroupIdx[i])[0]
        tmpMem = np.unique(tree[tt, 0:2]).astype(np.int)  #1D int64 array
        if nrObservation>0:
            tmpMem = tmpMem[tmpMem < nrObservation] #skip the member > size(transforms)
            if tmpMem.size == 0:
                tmpMem = np.array([], dtype = np.int)
        groups[i]["members"] = tmpMem  #member can be empty!
        groups[i]["tree"] = tree[tt,:]
        if groups[i]["members"].size != 0:
            for iii in groups[i]["members"]:
                dlabels[iii] = "c%d"%groups[i]["id"]
    
    link_cols = genLink_color(groups,tree)
    if dsp & (maxLeaves>0):
        plt.figure()
        if nrObservation > maxLeaves:
            figure_title = "clustering %d of %d shown"%(maxLeaves, nrObservation)
            plt.title(figure_title)
            with plt.rc_context({'lines.linewidth': 0.5}):
                dendrogram(tree,  p=maxLeaves, color_threshold=None,
                            link_color_func=lambda x: link_cols[x])
        else:
            figure_title  = 'clustering'
            plt.title(figure_title)
            with plt.rc_context({'lines.linewidth': 0.5}):
                dendrogram(tree,  p=maxLeaves, color_threshold=None,
                            link_color_func=lambda x: link_cols[x])
            plt.xticks(fontsize = 5)
            
    #add the legend
    if dsp & (len(groups) > 0) & (maxLeaves>0):
        h_plot = [ ]
        h_label = [ ]
        i = 0
        for single_dict in groups:
            if len(single_dict['members']) > 0:
                h_plot.append(plt.plot(1,1, color = single_dict['color']))
                i += 1
                h_label.append('cl:%d(%d)'%(single_dict['id'], len(single_dict['members'])))
        plt.legend(h_plot,labels = h_label,fontsize = 10,bbox_to_anchor=(1.15, 1),
                   title = 'class')  
        plt.tight_layout()   

    return groups, cmap, groupIdx, ColorThreshold
        
def genLink_color(groups, tree):
    #color dict for each leaf
    dflt_col = "#808080"   # Unclustered gray
    D_leaf_color = { }
    for single_cluster in groups:
        rgb = single_cluster['color']
        hexrgb = '#%02x%02x%02x'%(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        members =  single_cluster['members']       
        for single_member in members:        
            D_leaf_color["%d"%single_member] = hexrgb
    #color for each linkage
    link_cols = {}
    for i, i12 in enumerate(tree[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(tree) else D_leaf_color["%d"%x] for x in i12)
        link_cols[i+1+len(tree)] = c1 if c1 == c2 else dflt_col
        
    return link_cols
        
 
def genColorLookUp(Z, threshold): #Z:tree --- Array is changable data structure, You change it!!!
    if threshold == 'off':  #any transforms with different distance will be considered
        theGroups = ''
        groups = ''
        cmap = ''
    else:
        Z = transz(Z) #return the readable cluster formation tree
        cmap = np.array([1,0,0]) # 1-D
        numLeaves = Z.shape[0] + 1        
        groups = np.sum(Z[:,2] < threshold) #number of transform pairs considered
        if (groups > 1) & (groups < Z.shape[0]):
            theGroups = np.zeros(numLeaves-1,dtype = np.int32) #1-D array int class
            numColors = 0
            for count in np.arange(groups-1, -1, -1):
                if theGroups[count] == 0:
                    P = np.zeros(numLeaves-1,dtype = np.int32) #1D int array
                    P[count] = 1
                    P = colorcluster(Z,P,Z[count,0],count)
                    P = colorcluster(Z,P,Z[count,1],count)
                    numColors += 1
                    theGroups[P.astype(np.bool)] = numColors
            cmap = gen_colors(numColors)  #2D with numColorx3 #the hsv is different 
                                                               #with the default dendrogam
            cmap = np.concatenate((cmap,np.array([[0.7,0.7,0.7]])),axis = 0)
        else:
            theGroups = ''
            groups = ''
            cmap = ''
            
    return theGroups, groups, cmap
                
                
            
def colorcluster(X,T,k,m): #X, the tree
    '''
    find local clustering
    '''
    n = m
    while n > 0:
        n -= 1
        if X[n,0] == k: #node k is not a leave, it has subtree
            T = colorcluster(X,T,k,n)
            T = colorcluster(X,T,X[n,1],n)
            break
    T[m] = 1
    
    return T
            
        

def transz(Z):
    '''
    In the linkage function from scipy, thet named the newly formed cluster
    with index M+K, where M is the number of original observation, and K means
    the this new cluster is the kth clusterto be formed. This function converts
    the M+k indexing into min(i,j) indexing for newly formed clusters.
    '''
    numLeaves = Z.shape[0] + 1 #number of the obervations if only one new ob form cluster
    for i in range(numLeaves-1):
        if Z[i,0] > (numLeaves-1):
            Z[i,0] = traceback(Z,Z[i,0])
        if Z[i,1] > (numLeaves - 1):
            Z[i,1] = traceback(Z,Z[i,1])
        if Z[i,0] > Z[i,1]:
            Z[i,0:2] = Z[i, [1,0]] #change the position
    
    return Z
            
def traceback(Z,b):
    b = int(b)
    numLeaves = Z.shape[0] + 1
    if Z[b-numLeaves, 0] > (numLeaves-1):
        a = traceback(Z, Z[b-numLeaves,0])
    else:
        a = Z[b-numLeaves,0]
    if Z[b-numLeaves,1] > (numLeaves - 1):
        c = traceback(Z, Z[b-numLeaves,1])
    else:
        c = Z[b-numLeaves,1]
    
    return np.min([a,c])
        
    


def gen_colors(n): #one cluster one color
    colorm = sns.color_palette('hls',n)
    return np.array(colorm)

 

        