import os
import numpy as np
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import copy
import scipy.spatial.distance as ssd


from py_io.tom_starread import tom_starread, generateStarInfos
from py_io.tom_starwrite import tom_starwrite
from py_transform.tom_calcTransforms import tom_calcTransforms
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_cluster.tom_calcLinkage import tom_calcLinkage
from py_cluster.tom_dendrogram import tom_dendrogram
from py_cluster.tom_selectTransFormClasses import tom_selectTransFormClasses
from py_cluster.tom_A2Odist import tom_A2Odist
from py_cluster.tom_assignTransFromCluster import tom_assignTransFromCluster
from py_align.tom_align_transformDirection import tom_align_transformDirection
from py_link.tom_linkTransforms import tom_linkTransforms
from py_link.tom_find_poorBranch import tom_find_poorBranch
from py_link.tom_connectGraph import tom_connectGraph
from py_link.tom_find_transFormNeighbours import *
from py_summary.tom_analysePolysomePopulation import *
from py_summary.tom_analyseRiboAttrib import tom_analyseRiboAttrib
from py_summary.tom_genListFromTransForm import tom_genListFromTransForm
from py_summary.tom_genavgFromTransFormScript import tom_genavgFromTransFormScript
from py_stats.tom_kdeEstimate import tom_kdeEstimate
from py_vis.tom_plot_vectorField import tom_plot_vectorField
from py_mergePoly.tom_addTailRibo import tom_addTailRibo,saveStruct 
from py_log.tom_logger import Log



class Polysome:
    ''' 
    Polysome is a class that used to cluster neighbors and track linear assembles
    '''
    def __init__(self, input_star = None, run_time = 'run0', proj_folder = None, translist = None):
        '''
        set the default properties for polysome class
        input:
            inut_star: the star file for particles' euler angles and coordinates
            run_time: the times to get new linear assembles
        '''
        #io
        self.io = { }
        self.io['posAngList'] = input_star
        (filepath,tempfilename) = os.path.split(self.io['posAngList'])
        (shotname,extension) = os.path.splitext(tempfilename)
        if proj_folder is None:
            self.io['projectFolder'] = 'cluster-%s'%shotname
        else:
            self.io['projectFolder'] = proj_folder
        self.io['classificationRun'] = run_time 
        self.io['classifyFold'] = None  #should be modified only after run input
        
        #get transformations of neighbors  
        self.transForm = { }
        self.transForm['pixS'] = 3.42
        self.transForm['maxDist'] = 342 #the searching region of adjacent ribosomes
        self.transForm['branchDepth'] = 2 #searching the depeth of branch of ribosomes;0:clean branch
        
        #for classify
        self.classify = { }
        self.classify['clustThr'] = 0.12
        self.classify['cmb_metric'] = 'scale2Ang'
        
        #filter transform clusters
        self.sel = [ ]  
        self.sel.append({})
        self.sel[0]['clusterNr'] = np.array([-1])  #select the clusters of transform
        self.sel[0]['polyNr'] = np.array([-1])   #select the polysome (by polyID)
        self.sel[0]['list'] = 'Clusters-Sep' #select all clusters
      
        #vis
        self.vis  = { }
        self.vis["vectField"] = { }
        self.vis['vectField']['render'] = 1
        self.vis['vectField']['type'] = 'basic' # 'advance' ('basic':shows only lines and positions without repeat vectors)
        self.vis['vectField']['showTomo'] = np.array([-1])  #what tomos to vis?
        self.vis['vectField']['showClusterNr'] = np.arange(10000) #what clusters to vis?
        self.vis['vectField']['polyNr'] = np.array([-1]) #what polys to show (input polyIDs)?
        self.vis['vectField']['onlySelected'] = 1  #only vis those upper selected polysomes?
        self.vis['vectField']['repVect'] = np.array([[0,1,0]]) #how to represent the particles(angle + position) 
                                                               #must be 2D array. Rotate this vector to represent rotated particles
        self.vis['vectField']['repVectLen'] = 20
        
        #parameters to show the longest linear asseblems 
        self.vis['longestPoly'] = { }
        self.vis['longestPoly']['render'] = 1
        self.vis['longestPoly']['showClusterNr'] = np.array([-1]) #what clusters to vis longest linear asseblems?
        
        #for avg of Relion
        self.avg = { }
        self.avg['filt'] = { }
        self.avg['filt']['minNumPart'] = 100  #select number of transforms in each cluster
        self.avg['filt']['maxNumPart'] = np.inf  #select number of particles in each class       
        self.avg['pixS'] = self.transForm['pixS']
        self.avg['maxRes'] = 20 #needed by relion 
        self.avg['cpuNr'] = 0
        self.avg['callByPython'] = 0
        
        
        #for polysomes filling up 
        self.fillPoly = { }
        self.fillPoly['clusterNr'] = np.array([-1]) #which cluster of trans want to fillup?
        self.fillPoly['addNum'] = 1 #how many hypo particles at each end of linear asseblms?
        self.fillPoly['fitModel'] = 'genFit' #genFit:fit distribution based on data/lognorm:base on lognorm model
        self.fillPoly['threshold'] = 0.05 #threshold to accept filled up particles
    
        self.transList = translist
        
        #add logger
        logFile = '%s/%s'%(os.getcwd(), 'NEMO_TOC.log')  
        if os.path.exists(logFile):
            os.remove(logFile)
            
        self.log = Log('main').getlog()
        
    def creatOutputFolder(self):
        '''
        create output folders
        '''
        
        self.io['classifyFold'] = '%s/%s'%(self.io['projectFolder'],
                                           self.io['classificationRun'])
        projectFolder = self.io['projectFolder']
        classificationFolder = self.io['classifyFold']
        if not os.path.exists(projectFolder):
            os.mkdir(projectFolder)
        if not os.path.exists(classificationFolder):
            os.mkdir(classificationFolder)
            os.mkdir('%s/scores'%classificationFolder)

        #remove the dirs
        if  os.path.exists(classificationFolder): 
            if os.path.isdir('%s/stat'%classificationFolder):
                shutil.rmtree('%s/stat'%classificationFolder)
            if os.path.isdir('%s/clusters'%classificationFolder):
                shutil.rmtree('%s/clusters'%classificationFolder)            
            if os.path.isdir('%s/vis'%classificationFolder):
                shutil.rmtree('%s/vis'%classificationFolder)
         
        #create the dirs
        os.mkdir('%s/vis'%classificationFolder)
        os.mkdir('%s/vis/neighborsDist'%classificationFolder)
        os.mkdir('%s/vis/vectfields'%classificationFolder)
        os.mkdir('%s/vis/clustering'%classificationFolder)
        os.mkdir('%s/vis/innerClusterDist'%classificationFolder)
        os.mkdir('%s/vis/fitInnerClusterDist'%classificationFolder)
        os.mkdir('%s/vis/noiseEstimate'%classificationFolder)
        os.mkdir('%s/vis/averages'%classificationFolder)
        os.mkdir('%s/stat'%classificationFolder)
        os.mkdir('%s/clusters'%classificationFolder)
            
    def preProcess(self, if_stopgap, subtomoPath, ctfFile, minDist):
        '''
        remove the duplicates if two particles too close with each other,
        and change the stopgap file format into relion file format
        and recenter the particles
        '''
        self.log.info('Preprocess the input starfile.')
        _,fileName = os.path.split(self.io['posAngList'])
        posAngListInfo = tom_starread(self.io['posAngList'], self.transForm['pixS'])
        posAngList = posAngListInfo['data_particles']
        dataType = posAngListInfo['type']
        if dataType == 'stopgap': #from SG to Relion
            posAngListRelionInfo = self.sg2relion(posAngListInfo, subtomoPath, ctfFile)
            posAngListRelion = posAngListRelionInfo['data_particles']
        else:
            posAngListRelion = posAngList
        #recenter particles 
        if dataType == 'relion2' | dataType == 'stopgap':
            if 'rlnOriginX' in posAngListRelion.columns:
                posAngListRelion['rlnCoordinateX'] = posAngListRelion['rlnCoordinateX'] + posAngListRelion['rlnOriginX']
                posAngListRelion['rlnCoordinateY'] = posAngListRelion['rlnCoordinateY'] + posAngListRelion['rlnOriginY']
                posAngListRelion['rlnCoordinateZ'] = posAngListRelion['rlnCoordinateZ'] + posAngListRelion['rlnOriginZ']
            
        elif dataType == 'relion3':
            if 'rlnOriginXAngst' in posAngListRelion.columns:
                posAngListRelion['rlnCoordinateX'] = posAngListRelion['rlnCoordinateX'] + posAngListRelion['rlnOriginXAngst']/self.transForm['pixS']
                posAngListRelion['rlnCoordinateY'] = posAngListRelion['rlnCoordinateY'] + posAngListRelion['rlnOriginYAngst']/self.transForm['pixS']
                posAngListRelion['rlnCoordinateZ'] = posAngListRelion['rlnCoordinateZ'] + posAngListRelion['rlnOriginZAngst']/self.transForm['pixS']          
        #remove too closed neigbors
        uniq_mrc = np.unique(posAngListRelion['rlnMicrographName'].values)
        #remove the duplicates and also store the distance of two neighbors 
        rmIdx = [ ]
        distanceNeighborsB4RmDup = [ ]
        for single_mrc in uniq_mrc:
            posAngListRelionSingle = posAngListRelion[posAngListRelion['rlnMicrographName'] == single_mrc]
            posAngListRelionSingleArray = posAngListRelionSingle[['rlnCoordinateX',
                                                                  'rlnCoordinateY',
                                                                  'rlnCoordinateZ']].values
            posIdx = posAngListRelionSingle.index
            distances = ssd.pdist(posAngListRelionSingleArray, metric = 'euclidean')
            distancesMatrix = ssd.squareform(distances)
            #record the distance 
            for i in range(distancesMatrix.shape[0]):
                for j in range(i+1, distancesMatrix.shape[0]):    
                    distanceNeighborsB4RmDup.append(distancesMatrix[i][j])
                                   
            #bigger lowtri matrix
            minDistMatrix = np.ones((distancesMatrix.shape[0], distancesMatrix.shape[0]))*(minDist+100)
            minDistMatrix_lowTri = np.tril(minDistMatrix, k=0)
            distancesMatrix = distancesMatrix+minDistMatrix_lowTri
            rmPairi, rmPairj = np.where(distancesMatrix <= minDist)
            if len(rmPairi) == 0:
                continue
            for i, j in zip(rmPairi, rmPairj):
                if i == j:
                    continue
                else:                    
                    if 'rlnLogLikeliContribution' in posAngListRelionSingle.columns:
                        scorei = posAngListRelionSingle['rlnLogLikeliContribution'].values[i]
                        scorej = posAngListRelionSingle['rlnLogLikeliContribution'].values[j]
                        if scorei <= scorej:
                            rmIdx.append(posIdx[i])
                        else:
                            rmIdx.append(posIdx[j])
                    else:
                        rmIdx.append(posIdx[i])
  
        #unique            
        rmIdx = np.unique(rmIdx)
        if len(rmIdx) > 0:
            keepIdx = np.setdiff1d(np.asarray(posAngListRelion.index), rmIdx, assume_unique = True)                
        else:
            keepIdx = np.asarray(posAngListRelion.index)       
        fileNameNew = '%s/%s_recenter_rmDup.star'%(self.io['projectFolder'], fileName.split('.')[0])
        fileNameNewDrop = '%s/stat/%s_recenter_dup.star'%(self.io['classifyFold'], fileName.split('.')[0])
        
        posAngListRelion = posAngListRelion.iloc[keepIdx]
        posAngListRelion.reset_index(drop=True, inplace = True)
        
        #save drop particle above search radius 
        posAngListRelionDrop = posAngListRelion.iloc[rmIdx]
        posAngListRelionDrop.reset_index(drop=True, inplace = True)   
        
        #record the distances after duplicated remove
        distanceNeighborsAfterRmDup = [ ]
        for single_mrc in uniq_mrc:
            posAngListRelionSingle = posAngListRelion[posAngListRelion['rlnMicrographName'] == single_mrc]
            posAngListRelionSingleArray = posAngListRelionSingle[['rlnCoordinateX',
                                                                  'rlnCoordinateY',
                                                                  'rlnCoordinateZ']].values
            distances = ssd.pdist(posAngListRelionSingleArray, metric = 'euclidean')
            distancesMatrix = ssd.squareform(distances)
            #record the distance 
            for i in range(distancesMatrix.shape[0]):
                for j in range(i+1, distancesMatrix.shape[0]):    
                    distanceNeighborsAfterRmDup.append(distancesMatrix[i][j])    
                    
        #plot and store the distance distribution
        plt.figure()
        plt.hist(distanceNeighborsB4RmDup,bins = 50, label = 'before duplicates removal')
        plt.hist(distanceNeighborsAfterRmDup,bins = 50, label = 'after duplicates removal')
        plt.xlabel('Euclidean distances between the most neighbors(pixels)')
        plt.ylabel('# of distances')
        plt.savefig('%s/vis/neighborsDist/neighborsDistance.png'%self.io['classifyFold'], dpi = 300)
        plt.close()
            
        if if_stopgap:
            posAngListRelionInfo['data_particles'] = posAngListRelion
            tom_starwrite(fileNameNew, posAngListRelionInfo)
            
            posAngListRelionInfo['data_particles'] = posAngListRelionDrop
            tom_starwrite(fileNameNewDrop, posAngListRelionInfo)
            
            
        else:                                  
            posAngListInfo['data_particles'] = posAngListRelion            
            tom_starwrite(fileNameNew, posAngListInfo)
            
            posAngListInfo['data_particles'] = posAngListRelionDrop            
            tom_starwrite(fileNameNewDrop, posAngListInfo)            
            
        self.io['posAngList'] = fileNameNew    
            
    @staticmethod
    def sg2relion(starList, subtomoPath=None, ctfFile = None):
        relionStarInfo = generateStarInfos()
        relionStarInfo['pixelSize'] = starList['pixelSize']
        starFile = copy.deepcopy(starList['data_particles'])
        starFile.drop(['motl_idx', 'object', 'halfset'], axis = 1, inplace = True)
        cols = starFile.columns
        #transfer_dict 
        varDict = {'tomo_num':'rlnMicrographName', 'subtomo_num':'rlnImageName',
                   'orig_x':'rlnCoordinateX', 'orig_y':'rlnCoordinateY',
                   'orig_z':'rlnCoordinateZ', 'x_shift':'rlnOriginX',
                   'y_shift':'rlnOriginY','z_shift':'rlnOriginZ',
                   'class':'rlnClassNumber','score':'rlnLogLikeliContribution',
                   'phi':'rlnAngleRot','psi':'rlnAngleTilt','the':'rlnAnglePsi'}
        newCols = [varDict[i] for i in cols]
        starFile.columns = newCols
        #update ctf & pixel size
        starFile['rlnDetectorPixelSize'] = [relionStarInfo['pixelSize']]*starFile.shape[0]
        starFile['rlnCtfImage'] = [ctfFile]*starFile.shape[0]
        starFile['rlnMagnification'] = [10000]*starFile.shape[0]
        if 'rlnMicrographName' in starFile.columns:
            starFile['rlnMicrographName'] = ["tomo%d"%int(i) for i in starFile["rlnMicrographName"].values]
        if 'rlnImageName' in starFile.columns:
            starFile['rlnImageName'] = ["%s/subtomo_%d.mrc"%(subtomoPath,int(i)) for i in starFile["rlnImageName"].values]
        #update euler angles 
        rotList = np.zeros(starFile.shape[0])
        tiltList = np.zeros(starFile.shape[0])
        psiList = np.zeros(starFile.shape[0])
        for i in range(starFile.shape[0]):
            _, euler_out = tom_eulerconvert_xmipp(starFile['rlnAngleRot'].values[i], 
                                                                         starFile['rlnAngleTilt'].values[i], 
                                                                         starFile['rlnAnglePsi'].values[i],'tom2xmipp')
            rotList[i], tiltList[i], psiList[i] = euler_out  
        starFile['rlnAngleRot'] = rotList
        starFile['rlnAngleTilt'] = tiltList
        starFile['rlnAnglePsi'] = psiList
        
        relionStarInfo['data_particles'] = starFile
        
        return relionStarInfo
    
    
    def calcTransForms(self, worker_n = 1):
        '''
        generate the transformList
        worker_n: the number of cpus to process
        '''
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        transFormFile = '%s/%s'%(self.io['classifyFold'],'allTransforms.star')

        if os.path.exists(transFormFile):
            self.log.info('Load distances from %s'%transFormFile)
            self.transList = tom_starread(transFormFile, self.transForm['pixS'])
            self.transList = self.transList['data_particles']
        else:
            self.transList = tom_calcTransforms(self.io['posAngList'], self.transForm['pixS'], maxDistInPix, '',
                                                'exact', transFormFile, 1, worker_n)
    
    def groupTransForms(self, worker_n = 1, iterN = 1, gpu_list = None, freeMem = None):
        '''
        generate the clustering transform clusters
        '''
        if gpu_list is not None:
            worker_n = None
        self.log.info('Start clustering')
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        outputFold = '%s/scores'%self.io['classifyFold']
        treeFile = '%s/tree.npy'%outputFold
        ###if do clustering in a small subset 
        if self.transList.shape[0] > 100000:
            self.log.info('The number of transforms is to big, will cluster top 100,000 transforms.')           
            transList_subset = self.transList.iloc[0:100000, :]
        else:
            transList_subset = self.transList
      
        if os.path.exists(treeFile):
            self.log.info('load tree from %s'%treeFile)
            ll = np.load(treeFile) #load the clustering tree models 
        else:
            ll = tom_calcLinkage(transList_subset, outputFold, maxDistInPix,
                                 self.classify['cmb_metric'], worker_n, 
                                 gpu_list, freeMem) 
        
        clusters, _, _, thres = tom_dendrogram(ll, self.classify['clustThr'], transList_subset.shape[0], 0, 0)        
        self.classify['clustThr'] = thres  
        
        if len(clusters) == 0:
            self.log.warning('''No clusters! Check the threshold you input! You put 
                                a very high/low threshold.''')

        else:
            #this step give the cluster id for each transform
            for single_dict in clusters:
                idx = single_dict['members']
                classes = single_dict['id']
                if len(idx) == 0: #no cluster detect 
                    continue
                colour = "%.2f-%.2f-%.2f"%(single_dict['color'][0],
                                           single_dict['color'][1],
                                           single_dict['color'][2])
                transList_subset.loc[idx, "pairClass"] = classes
                transList_subset.loc[idx, 'pairClassColour'] = colour
        if transList_subset.shape[0] < self.transList.shape[0]:
            #first aligned the transforms and summary it!
            transList_subset = tom_align_transformDirection(transList_subset)
            allClusters = transList_subset['pairClass'].values
            allClustersU = np.unique(allClusters)
            cmb_metric = self.classify['cmb_metric']
            stats_cluster = { }
            for single_cluster in allClustersU:
                if single_cluster == 0:#no need analysis cluster0
                    continue
                idx = np.where(allClusters == single_cluster)[0]      
                vectStat, distsVect = calcVectStat(transList_subset.iloc[idx,:])
                angStat, distsAng = calcAngStat(transList_subset.iloc[idx,:])           
                if cmb_metric == 'scale2Ang':
                    distsVect2 = distsVect/(2*maxDistInPix)*180
                    distsCN = (distsAng+distsVect2)/2
                elif cmb_metric == 'scale2AngFudge':
                    distsVect2 = distsVect/(2*maxDistInPix)*180
                    distsCN = (distsAng+(distsVect2*2))/2
                    
                stats_cluster[single_cluster] = np.zeros(7)
                stats_cluster[single_cluster][0] = np.max(distsCN)
                stats_cluster[single_cluster][1:4] = [vectStat['meanTransVectX'], vectStat['meanTransVectY'], vectStat['meanTransVectZ']]
                stats_cluster[single_cluster][4:7] = [angStat['meanTransAngPhi'], angStat['meanTransAngPsi'], angStat['meanTransAngTheta']]
                
            pairClassList, _ = tom_assignTransFromCluster(self.transList, stats_cluster, cmb_metric, maxDistInPix, iterN)
            #update the pairClass as well as the color 
            clusterColor = { }
            for sgCluster, sgColor in zip(transList_subset['pairClass'].values, transList_subset['pairClassColour'].values):
                clusterColor[sgCluster] = sgColor
            pairClassColorList = [ ]
            for sgCluster in pairClassList:
                pairClassColorList.append(clusterColor[sgCluster])
            self.transList["pairClass"] = pairClassList
            self.transList['pairClassColour'] = pairClassColorList
                       
        self.log.info('Clustering done')
      
    def selectTransFormClasses(self, worker_n = 1, gpu_list = None, itrClean = 1):
        '''
        select any cluster and Relink
        itrClean: # of cycles to clean the data 
        '''
        transListSel = ''
        selFolds = ''
        if self.sel[0]['minNumTransform'] != -1:
            self.sel[0]['minNumTransform'] = int(self.sel[0]['minNumTransform']*self.transList.shape[0]) #default is 1%
            self.log.info('set #minTransform as %d to select clusters'%self.sel[0]['minNumTransform'])
                          
    
            for _ in range(itrClean):  
                #this step can keep classes we want for further analysis as well as remove cluster 0                 
                _,_, transListSelCmb = tom_selectTransFormClasses(self.transList,
                                                                  self.sel[0]['list'],
                                                                  self.sel[0]['minNumTransform'], '')
                if len(transListSelCmb) == 0:
                    errorInfo = """No transform members kept after cleaning. Try to reduce   
                                   parameter:minNumTransformPairs and try again!"""
                    self.log.error(errorInfo)
                    raise TypeError(errorInfo)
                    
                self.transList = transListSelCmb
                #this select can discard the transforms with class ==0 (which failed to form cluster)
                os.rename('%s/scores/tree.npy'%self.io['classifyFold'],
                         '%s/scores/treeb4Relink.npy'%self.io['classifyFold'])
                self.groupTransForms(worker_n, gpu_list)
                
            if os.path.exists('%s/allTransforms.star'%self.io['classifyFold']):
                os.rename('%s/allTransforms.star'%self.io['classifyFold'],
                              '%s/allTransformsb4Relink.star'%self.io['classifyFold'])
                
            #store the translist
            starInfo = generateStarInfos()
            starInfo['data_particles'] = self.transList
            tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], starInfo) 
                
        transListSel, selFolds, _ = tom_selectTransFormClasses(self.transList,
                                                               self.sel[0]['list'],
                                                               self.sel[0]['minNumTransform'],
                                                               '%s/clusters'%(self.io['classifyFold']))  
           
        if len(transListSel) == 0:
            self.log.warning('''No translist has been selected for further analysis!
                                Try to reduce minNumTransformPairs and try again!''')
     
        return transListSel, selFolds  #transListSel be empty when on clustering performs  
    
    def alignTransforms(self):
        '''
        In each cluster, align the direction of each transform to the same direction.
        '''
        warnings.filterwarnings('ignore')
        self.log.info('Align transform pairs')
        self.transList = tom_align_transformDirection(self.transList)
        
        #store the translist
        starInfo = generateStarInfos()
        starInfo['data_particles'] = self.transList
        tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], starInfo) 
        
        self.log.info('Align transform pairs done')
         
    def find_connectedTransforms(self, allClassesU = None, saveFlag = 1, 
                                 worker_n = 1, gpu_list = None):
       
        '''
        track the polyribosomes
        the branch analysis is still developing
        '''
        warnings.filterwarnings('ignore')
        self.log.info('Track polysomes')

        cmb_metric = self.classify['cmb_metric']
        pruneRad = self.transForm['maxDist']/self.transForm['pixS']
        allClasses = self.transList['pairClass'].values
        if allClassesU is None:
            allClassesU = np.unique(allClasses)        
        allTomos = self.transList['pairTomoID'].values
        allTomosU = np.unique(allTomos)
        
        
        if self.transForm['branchDepth'] == 0: #clean branch 
            self.log.info('Clean branches')
            #load the information of avgshift/rot           
            classSummaryList = '%s/stat/statPerClass.star'%self.io['classifyFold']
            if os.path.exists(classSummaryList):
                classSummaryList = tom_starread(classSummaryList)
                classSummaryList = classSummaryList['data_particles']
            
            #clean the branch in specific classes
            idx_rm = np.array([], dtype = np.int)
            for single_class in allClassesU:
                if single_class == 0:
                    continue
                idx = np.where(allClasses == single_class)[0]
                #calculate the average shift and rotation for each class
                if isinstance(classSummaryList, str):
                    vectStat, _ =  calcVectStat(self.transList.iloc[idx,:])
                    angStat, _  =  calcAngStat(self.transList.iloc[idx,:])
                    #extract the shift and rot information
                    avgShift = np.array([vectStat['meanTransVectX'], vectStat['meanTransVectY'], 
                                         vectStat['meanTransVectZ']])
                    avgRot  =  np.array([angStat['meanTransAngPhi'], angStat['meanTransAngPsi'], 
                                         angStat['meanTransAngTheta']])
                else:
                    avgShift = classSummaryList[classSummaryList['classNr'] == single_class].loc[:,
                                       ['meanTransVectX','meanTransVectY','meanTransVectZ']].values[0] 
                    avgRot = classSummaryList[classSummaryList['classNr'] == single_class].loc[:,
                                       ['meanTransAngPhi','meanTransAngPsi','meanTransAngTheta']].values[0] 
                    
                idx_drop = tom_find_poorBranch(self.transList.iloc[idx,:], avgShift, 
                                               avgRot, worker_n, gpu_list, cmb_metric, 
                                               pruneRad) #this function can find the index with branch, which will be removed!
                if len(idx_drop) > 0:
                    self.log.info('Find %d branches in clusters:%d, will be removed!'%(len(idx_drop),single_class))
                    idx_rm = np.concatenate((idx_rm, idx_drop))
                
            if len(idx_rm) > 0:
                idxPair12_rm =  self.transList.loc[idx_rm, ['pairIDX1', 'pairIDX2']].values 
                self.transList = self.transList.drop(index = idx_rm)
                #update index 
                self.transList = self.transList.reset_index(drop=True)
                #update allClasses and allTomos
                allClasses = self.transList['pairClass'].values
                allClassesU_update = np.unique(allClasses)
                allClassesU = np.intersect1d(allClassesU_update, allClassesU, assume_unique = True)
                allTomos = self.transList['pairTomoID'].values
                allTomosU = np.unique(allTomos)
                
        else:
            self.log.info('Skip branch cleanning!')
                    
        br = { } #1D-array, check the existence of branches
        for single_class in allClassesU:           
            if single_class == 0:  #also should aviod single_class == -1
                continue  ##no cluster occur with class == 0  
            if single_class == -1:
                self.log.warning('''No clusters detected.
                                    ==> highly suggest do groupTransform before this step''')
            br[single_class] = 0  
            idx1 = np.where(allClasses == single_class)[0]
            offset_PolyID = 0
            count = 0
            for single_tomo in allTomosU:
                idx2 = np.where(allTomos == single_tomo)[0]
                idx = np.intersect1d(idx1, idx2)
                #track the polysomes in the same tomogram and the same transform class
                if len(idx) > 1:
                    self.transList.loc[idx,:], _, ifBranch, offset_PolyID = tom_linkTransforms(self.transList.iloc[idx,:],
                                      self.transForm['branchDepth'], offset_PolyID)
                    count += ifBranch
                elif len(idx) == 1:
                    offset_PolyID += 1
                    self.transList.loc[idx,'pairLabel'] = offset_PolyID  #begin with 1
                    self.transList.loc[idx,'pairPosInPoly1'] = 1 #the order in each polysome, begin with 1
                    self.transList.loc[idx,'pairPosInPoly2'] = 2 #the order in branch, ~= poly1 +1 
                
            br[single_class] = count
            

        class_branch = [key for key in br.keys() if br[key] > 0]
        if len(class_branch) > 0:
            self.log.warning('''Warning: branches in these cluster: %s
                                ==>can try to make smaller clusters'''%(str(class_branch)))

        self.log.info('Polysome tracking done')
     
        if saveFlag:
            starInfo = generateStarInfos()
            starInfo['data_particles'] = self.transList
            tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], starInfo) 
            
        if self.transForm['branchDepth'] == 0:
            if len(idx_rm) >0:
                return idxPair12_rm
        else:
            return None
        
    def analyseTransFromPopulation(self,  outputFolder = '', visFolder = '', if_summary = 1, verbose = 1):
        '''
        summary each transforms class, like if has branch/the length of the polysome
        all is saved at the stat folder 
        '''
        self.log.info('Summary transform classes and polysomes')
        if outputFolder == '':
            outputFolder = '%s/stat'%self.io['classifyFold']
        if visFolder == '':
            visFolder = '%s/vis'%self.io['classifyFold']
        maxDistInpix = self.transForm['maxDist']/self.transForm['pixS']
        cmb_metric = self.classify['cmb_metric']
        
        allClasses = self.transList['pairClass']
        allClassesU = np.unique(allClasses)
        stat = [ ]
        statPerPolyTmp = [ ]
        for single_class in allClassesU:
            if single_class == 0:#no need analysis class0
                continue
            idx = np.where(allClasses == single_class)[0]
            stat.append(analysePopulation(self.transList.iloc[idx,:], 
                                          maxDistInpix, visFolder, cmb_metric))
            statPerPolyTmp.append(analysePopulationPerPoly(self.transList.iloc[idx,:]))
        
        
        statPerPoly = sortStatPoly(statPerPolyTmp)
        stat = sortStat(stat)
        writeOutputStar(stat, outputFolder, label = 'cluster')
        writeOutputStar(statPerPoly, outputFolder, label = 'poly')
        if verbose:
            genOutput(stat, minTransMembers = 10)
        
        if if_summary:
            transListB4Relink = '%s/allTransformsb4Relink.star'%self.io['classifyFold']
            #transListB4Relink = '%s/allTransformsFillUp.star'%self.io['classifyFold']
            if not os.path.exists(transListB4Relink):
                transListB4Relink = None
            #analysis the overlapping of different transform classes
            tom_analyseRiboAttrib(self.transList, outputFolder,
                                  transListB4Relink, self.io['posAngList'])    
               
    def genOutputList(self, transListSel, outputFoldSel):
        '''
        output the summary of polysomes of each transform class.
        ATTENTION: the transListSel in this script is without any polysome ID info!!
        '''
        self.log.info('Generate selection translists')       
        for i in range(len(transListSel)):  #each translist represents one translist of one class
            transListTmp = transListSel[i]
            outputFoldCenter = "%s/pairCenter/"%outputFoldSel[i]
            tom_genListFromTransForm(transListTmp, outputFoldCenter, 'center')
            outputFoldCenter = "%s/particleCenter/"%outputFoldSel[i]
            tom_genListFromTransForm(transListTmp, outputFoldCenter, 'particle')

        
    def generateTrClassAverages(self):
        '''
        use relion to average the ribosome from each transform class
        '''
        if np.isinf(self.avg['filt']['minNumPart']):
            self.log.info('Skip translational class density map averaging')
            return 
        wk = "%s/clusters/c*/particleCenter/allParticles.star"%self.io['classifyFold']
        outfold = '%s/avg/exp/%s/c'%(self.io['classifyFold'], self.io['classificationRun'])
        
        tom_genavgFromTransFormScript(wk, self.avg['maxRes'], self.avg['pixS'],
                                       self.avg['cpuNr'], self.avg['filt'],
                                       self.avg['callByPython'],outfold)
           
              
    def link_ShortPoly(self, remove_branch = 1, worker_n = 1):
        '''
        link shorter polysomes 
        logic: put ribos at end of each polysome, and judge if added ribosomes 
        can link the head ribosome of other polysomes
        '''
        classList = np.unique(self.transList['pairClass'].values)
        self.transList['fillUpProb'] = -1*np.ones(self.transList.shape[0])
        if self.fillPoly['addNum'] == 0:
            self.log.info('skip polysome filling up')
            if remove_branch:
                self.transForm['branchDepth'] = 0
            self.find_connectedTransforms(classList, 0)
            saveStruct('%s/allTransformsFillUp.star'%self.io['classifyFold'], self.transList)
            return
    
        #load summary star file
        classSummaryList = '%s/stat/statPerClass.star'%self.io['classifyFold']
        if os.path.exists(classSummaryList):
            classSummary = tom_starread(classSummaryList)
            classSummary = classSummary['data_particles']
        else:
            self.log.error('lack file:%s, should run analyseTransFromPopulation!'%classSummaryList)
            return
        
        #using networkx(tom_connectGraph) to find the ribosomes to link OR to be linked
        statePolyAll_forFillUp = tom_connectGraph(self.transList)
         
        #give the store name & path of the particle star and transList
        (_,tempfilename) = os.path.split(self.io['posAngList'])
        (shotname,extension) = os.path.splitext(tempfilename)
        transListOutput = '%s/allTransformsFillUp.star'%self.io['classifyFold']
        particleOutput = '%s/%sFillUp.star'%(self.io['classifyFold'], shotname)
        
        #the method to accept fillUped ribosomes
        method = self.fillPoly['fitModel'] 
        self.log.info('link polys using %s'%method)
        
        for pairClass in classList:
            if pairClass == -1:
                self.log.warning('can not detect transformation cluster!')
                return
            if pairClass == 0:
                continue    
            
            #check if exist filled up ribosomes.If exist, then load it
            if os.path.exists(particleOutput):
                readPosAng = particleOutput
            else:
                readPosAng = self.io['posAngList']
            statePolyAll_forFillUpSingleClass = statePolyAll_forFillUp[statePolyAll_forFillUp['pairClass'] == pairClass ]          
            #read the avgshift and avgRot 
            avgShift = classSummary[classSummary['classNr'] == pairClass].loc[:,
                                   ['meanTransVectX','meanTransVectY','meanTransVectZ']].values[0] #1D array
            avgRot = classSummary[classSummary['classNr'] == pairClass].loc[:,
                                   ['meanTransAngPhi','meanTransAngPsi','meanTransAngTheta']].values[0] #1D array
                       
            cmbDistMaxMeanStd = ( classSummary[classSummary['classNr'] == pairClass]['maxCNDist'].values[0],
                                  classSummary[classSummary['classNr'] == pairClass]['meanCNDist'].values[0],
                                  classSummary[classSummary['classNr'] == pairClass]['stdCNDist'].values[0])
                
        
            transNr = self.transList[self.transList['pairClass'] == pairClass].shape[0]
            
            if (transNr < 50) & (method != 'max'):
                self.log.warning('only %d transform in cluster%d, sugget using max method for ribosomes fillingUp!'%(transNr, pairClass))
                   
            self.transList = tom_addTailRibo(statePolyAll_forFillUpSingleClass, self.transList, pairClass, avgRot, 
                                             avgShift, cmbDistMaxMeanStd,
                                             readPosAng, self.transForm['maxDist']/self.transForm['pixS'],                                         
                                             transListOutput, particleOutput,
                                             self.fillPoly['addNum'], 
                                             0, method,
                                             self.fillPoly['threshold'], worker_n)                

        self.log.info('polysome filling up done')
        #retrack the polysomes 
        self.transList['pairLabel'] = -1
        self.transList['pairPosInPoly1'] = -1
        self.transList['pairPosInPoly2'] = -1
        if remove_branch:
            self.transForm['branchDepth'] = 0 #clean the branches
        self.find_connectedTransforms(classList, 0)
        #save the transList 
        saveStruct(transListOutput,self.transList)
        
        
    def noiseEstimate(self):
        '''
        this method estimates the errors of transform classes assignment.
        First, calculating the distance between each transform from classA and the average
        transform of classA. Then calculate the distance between each transform
        aren't from classA and the average transform of classA.
        Finally, compare these two distributions.
        '''
        self.log.info('Estimate clusters assignment errors')
        
        maxDistInpix = self.transForm['maxDist']/self.transForm['pixS']
        cmb_metric = self.classify['cmb_metric']
        #figure saving directory 
        save_dir = '%s/vis/noiseEstimate'%self.io['classifyFold']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #load main files
        transList = '%s/allTransforms.star'%self.io['classifyFold']            
        transList = tom_starread(transList)
        transList = transList['data_particles']  
        #load class0 information
        transb4Relink = '%s/allTransformsb4Relink.star'%self.io['classifyFold']
        if os.path.exists(transb4Relink):
            transb4Relink = tom_starread(transb4Relink)
            transb4Relink = transb4Relink['data_particles']
        
        #load summary star file
        statSummary = '%s/stat/statPerClass.star'%self.io['classifyFold']
        if os.path.exists(statSummary):
            statSummary = tom_starread(statSummary)
            statSummary = statSummary['data_particles'] 
        else:
            errorInfo = '%s is missing'%statSummary
            self.log.error(errorInfo)
            raise FileNotFoundError(errorInfo)
        #find out the class0  transform
        keep_idx = [ ]
        if not isinstance(transb4Relink, str):
            transClass0 = ''
            for single_row in range(transb4Relink.shape[0]):
                idx1, idx2 = transb4Relink['pairIDX1'].values[single_row], \
                             transb4Relink['pairIDX2'].values[single_row]
                pair1 = transList[(transList['pairIDX1'] == idx1) & (transList['pairIDX2'] == idx2)].shape[0]
                pair2 = transList[(transList['pairIDX2'] == idx1) & (transList['pairIDX1'] == idx2)].shape[0]
                if (pair1+pair2) > 0:
                    continue
                keep_idx.append(single_row)
                
            transClass0 = transb4Relink.iloc[keep_idx,:]
        else:
            transClass0 = ''
        #calculte the distance between same class & different class   
        
        for single_row in range(statSummary.shape[0]):
            classN = statSummary['classNr'].values[single_row]
            avgShift = statSummary.loc[single_row,['meanTransVectX',
                                                   'meanTransVectY',
                                                   'meanTransVectZ']].values #1D array
            avgRot = statSummary.loc[single_row,['meanTransAngPhi',
                                                 'meanTransAngPsi',
                                                 'meanTransAngTheta']].values
            #for distance calculation from the same class
            transVectSame = transList[transList['pairClass'] == classN].loc[:,['pairTransVectX',
                                                                    'pairTransVectY','pairTransVectZ']].values #2D array
            transRotSame = transList[transList['pairClass'] == classN].loc[:,['pairTransAngleZXZPhi',
                                                                    'pairTransAngleZXZPsi','pairTransAngleZXZTheta']].values
    
            distVectSame, distAngSame, distCombineSame = tom_A2Odist(transVectSame, transRotSame, 
                                                                     avgShift, avgRot,                                                         
                                                                     1, None, cmb_metric, maxDistInpix)
            
            #for distance calculation from different class
            transVectDiff = transList[transList['pairClass'] != classN].loc[:,['pairTransVectX',
                                                                  'pairTransVectY','pairTransVectZ']].values    
            transRotDiff = transList[transList['pairClass'] != classN].loc[:,['pairTransAngleZXZPhi',
                                                                    'pairTransAngleZXZPsi','pairTransAngleZXZTheta']].values
            if not isinstance(transClass0, str):
                transVectDiff = np.concatenate((transVectDiff,                                     
                                    transClass0.loc[:,['pairTransVectX',
                                                       'pairTransVectY',
                                                       'pairTransVectZ']].values),axis = 0)
                transRotDiff = np.concatenate((transRotDiff,
                                      transClass0.loc[:,['pairTransAngleZXZPhi',
                                                         'pairTransAngleZXZPsi',
                                                         'pairTransAngleZXZTheta']]),axis = 0)
            distVectDiff, distAngDiff, distCombineDiff = tom_A2Odist(transVectDiff, transRotDiff, 
                                                                   avgShift, avgRot, 
                                                                   1, None, cmb_metric, maxDistInpix)

            #call the fit of KDE function       
            tom_kdeEstimate(distVectSame, 'Cluster %d'%classN, 'vect distance', save_dir,1, 0.05, distVectDiff, 'other clusters')
            tom_kdeEstimate(distAngSame, 'Cluster %d'%classN, 'angle distance', save_dir,1, 0.05, distAngDiff, 'other clusters')
            tom_kdeEstimate(distCombineSame, 'Cluster %d'%classN,'',save_dir,1, 0.05, distCombineDiff, 'other clusters') 
            #tom_kdeEstimate(distCombineSame, 'inner-cluster','',save_dir,1, 0.05, distCombineDiff, 'inter-clusters') 


        self.log.info('Error estimation done')
    
    def visResult(self):
        '''
        visulize the polysomes 
        as well as the linkage results
        '''
        self.log.info('Render figure')
        vectVisP = self.vis['vectField']
        if vectVisP['render']:
            tom_plot_vectorField(self.transList, vectVisP['type'], vectVisP['showTomo'], 
                                  vectVisP['showClassNr'], vectVisP['polyNr'], 
                                  vectVisP['onlySelected'],vectVisP['repVectLen'],vectVisP['repVect'],
                                  np.array([0.7,0.7,0.7]), '%s/vis/vectfields'%self.io['classifyFold'])
        else:
            self.log.info('VectorFiled rendering skipped')              
        treeFile = '%s/scores/tree.npy'%self.io['classifyFold']
        thres = self.classify['clustThr']
        
        self.dspTree(self.io['classifyFold'], treeFile, self.classify['clustThr'], -1)
        self.dspLinkage(self.io['classifyFold'], treeFile, thres)
            
    @staticmethod
    def dspTree(classifyFold, treeFile, clustThr, nrTrans=-1):             
        _, _, _, _ = tom_dendrogram(treeFile, clustThr, nrTrans, 1, 500)
        
        plt.ylabel('linkage score')        
        plt.savefig('%s/vis/clustering/tree.png'%classifyFold, dpi = 300)
        plt.close()
        
    @staticmethod
    def dspLinkage(classifyFold, treeFile,thres):
        plt.figure()
        plt.title('link-levels')
        tree = np.load(treeFile)
        plt.plot(np.sort(tree[:,2])[::-1], label = 'link-level')
        plt.plot(np.sort(tree[:,2])[::-1], 'ro', label = 'link-level')
        plt.plot(np.ones(tree.shape[0])*thres, 'k--' ,label = 'threshold')
        plt.legend()
        plt.text(1,thres*2, 'threshold = %.2f'%thres)
        plt.xlabel('# of transforms pairs')
        plt.ylabel('linkage score')
        
        plt.savefig('%s/vis/clustering/linkLevel.png'%classifyFold, dpi = 300)
        plt.close()
        
    def visLongestPoly(self):
        '''
        this method is aimed to vis longest polysomes 
        of each transform class
        '''
        polyVisP = self.vis['longestPoly']
        if polyVisP['render'] == 0:
            self.log.info('Render figures done')
            return 
        else:
            self.log.info('Rendering longest polysomes')
            showClassNr = polyVisP['showClassNr']
            if showClassNr[0] < 0:
                showClassNr = np.unique(self.transList['pairClass'].values)
            for singleClass in showClassNr:
                if singleClass == 0:
                    continue
                
                transList_singleClass = self.transList[self.transList['pairClass'] == singleClass]
                transList_singleClass['pairLabel_fix'] = np.fix(transList_singleClass['pairLabel'].values)
                polyLenList = transList_singleClass['pairLabel_fix'].value_counts()
                #show the distribution of polysome length
                plt.hist(polyLenList.values+1,bins=20)
                plt.xlabel('# of ribosomes in each polysome')
                plt.ylabel('# of polysomes')
                plt.savefig('%s/vis/vectfields/c%d_polyLengthDist.png'%(self.io['classifyFold'],singleClass),dpi = 300)
                plt.close()
                #find the longest polysome              
                longestPolyId = polyLenList.index[0]
                keep_row = np.where(transList_singleClass['pairLabel_fix'] == longestPolyId)[0]    
                transListLongestPoly = transList_singleClass.iloc[keep_row, :]
                #plot the longest polysome
                tom_plot_vectorField(posAng = transListLongestPoly, mode= self.vis['vectField']['type'],  
                                     outputFolder = '%s/vis/vectfields/c%d_longestPoly.png'%(self.io['classifyFold'],singleClass)
                                     ,if_2views = 1) 
                del transListLongestPoly
                self.log.info('Render figures done')
     

