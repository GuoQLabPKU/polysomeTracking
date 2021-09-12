import os
import numpy as np
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import timeit as ti
import pandas as pd


from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from py_transform.tom_calcTransforms import tom_calcTransforms
from py_cluster.tom_calcLinkage import tom_calcLinkage
from py_cluster.tom_dendrogram import tom_dendrogram
from py_cluster.tom_selectTransFormClasses import tom_selectTransFormClasses
from py_align.tom_align_transformDirection import tom_align_transformDirection
from py_link.tom_linkTransforms import tom_linkTransforms
from py_link.tom_find_poorBranch import tom_find_poorBranch
from py_link.tom_find_transFormNeighbours import *
from py_summary.tom_analysePolysomePopulation import *
from py_summary.tom_findPattern import tom_findPattern
from py_summary.tom_genListFromTransForm import tom_genListFromTransForm
from py_summary.tom_avgFromTransForm import tom_avgFromTransForm
from py_summary.genMapVisScript import genMapVisScript
from py_vis.tom_plot_vectorField import tom_plot_vectorField
from py_link.tom_connectGraph import tom_connectGraph
from py_mergePoly.tom_addTailRibo import tom_addTailRibo,saveStruct 



class Polysome:
    ''' 
    Polysome is one class that used to track the polysomes in
    the ribosome groups
    '''
    def __init__(self, input_star = None, run_time = 'run0',translist = None):
        '''
        set the default properties for polysome class
        input:
            inut_star: the star file for ribosomes euler angles and coordinates
            run_time: the times to get new polysomes
        '''
        #for io
        self.io = { }
        self.io['posAngList'] = input_star
        (filepath,tempfilename) = os.path.split(self.io['posAngList'])
        (shotname,extension) = os.path.splitext(tempfilename)
        self.io['projectFolder'] = 'cluster-%s'%shotname
        self.io['classificationRun'] = run_time 
        self.io['classifyFold'] = None  #should be modify only after run input
        
        #for geting transformation of ribosome pairs and polysome
        self.transForm = { }
        self.transForm['pixS'] = 3.42
        self.transForm['maxDist'] = 342 #the searching region of adjacent ribosomes
        self.transForm['branchDepth'] = 2 #searching the depeth of branch of ribosomes;using 0 to clean branch
        
        #for classify
        self.classify = { }
        self.classify['clustThr'] = 0.12
        self.classify['relinkWithoutSmallClasses'] = 1
        self.classify['cmb_metric'] = 'scale2Ang'
        
        #for select the clean polysomes
        self.sel = [ ]  #shoule be a list
        self.sel.append({})
        self.sel[0]['classNr'] = np.array([-1])  #select the class of transform
        self.sel[0]['polyNr'] = np.array([-1])   #select the polysome (by polyID)
        self.sel[0]['list'] = 'Classes-Sep' #select the whole classes 
        #we can add more requirement to select the clusters/polysome
        
        #for vis
        self.vis  = { }
        self.vis["vectField"] = { }
        self.vis['vectField']['render'] = 1
        self.vis['vectField']['showTomo'] = np.array([-1])  #what tomos to vis?
        self.vis['vectField']['showClass'] = np.arange(10000) #what classes of trans to vis?
        self.vis['vectField']['onlySelected'] = 1  #only vis those selected by upper params?
        self.vis['vectField']['polyNr'] = np.array([-1]) #what polys to show (input polyIDs)?
        self.vis['vectField']['repVect'] = np.array([[0,1,0]]) #how to represents the ribosomes(angle + position)
                                                               #must be 2D array
        self.vis['vectField']['repVectLen'] = 20 
        
        #for avg
        self.avg = { }
        self.avg['command'] = 'tom'
        self.avg['filt'] = { }
        self.avg['filt']['minNumTransform'] = np.inf  #select number of transforms in each class 
        self.avg['filt']['maxNumPart'] = 500  #select the particles in each class
        self.avg['pixS'] = 3.42
        self.avg['maxRes'] = 45 #params needed by relion 
        
        #for fw, create the forward model(rotate mrc and shift)
        self.fw = { } 
        self.fw['Map'] = 'vol4forward.mrc'
        self.fw['minNumTransform'] = 0 #threshold for transform class to generate formodel
        self.fw['pixS'] = self.transForm['pixS']
        
        #for Conf class Analysis
        self.clSt = { }
        self.clSt['findPat'] = { }
        self.clSt['findPat']['classNr'] =  np.array([-2])  #try to make this an array #-2
        self.clSt['findPat']['filt'] = { }
        self.clSt['findPat']['filt']['operator'] = '>'
        self.clSt['findPat']['filt']['value'] = 3
        
        #for polysomes filling up 
        self.fillPoly = { }
        self.fillPoly['class'] = np.array([-2]) #which class of trans want to relink?
        self.fillPoly['riboinfo'] = 1 #if print out the infos of filled up ribos(0 to switch off)
        self.fillPoly['addNum'] = 1 #how many hypo ribosomes at each end of polysome?
        
        #add the transformation data 
        self.transList = translist
        
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
        if  os.path.exists(classificationFolder): #it seems that only scores dir kept ans allTransforms.star
            if os.path.isdir('%s/stat'%classificationFolder):
                shutil.rmtree('%s/stat'%classificationFolder)
            if os.path.isdir('%s/classes'%classificationFolder):
                shutil.rmtree('%s/classes'%classificationFolder)        
            if os.path.isdir('%s/avg'%classificationFolder):
                shutil.rmtree('%s/avg'%classificationFolder)         
            if os.path.isdir('%s/vis'%classificationFolder):
                shutil.rmtree('%s/vis'%classificationFolder)
         
        
        os.mkdir('%s/vis'%classificationFolder)
        os.mkdir('%s/vis/vectfields'%classificationFolder)
        os.mkdir('%s/vis/clustering'%classificationFolder)
        os.mkdir('%s/vis/averages'%classificationFolder)
        os.mkdir('%s/vis/distVSavg'%classificationFolder)
        os.mkdir('%s/vis/fitDist'%classificationFolder)
        os.mkdir('%s/stat'%classificationFolder)
        os.mkdir('%s/avg'%classificationFolder)
        os.mkdir('%s/avg/exp'%classificationFolder)
        os.mkdir('%s/avg/model'%classificationFolder)
        os.mkdir('%s/classes'%classificationFolder)
            
            
    def calcTransForms(self, worker_n = 1):
        '''
        give the input ribosomes input star,
        generate the transformList
        worker_n: the number of cpus to process
        '''
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        transFormFile = '%s/%s'%(self.io['classifyFold'],
                                 'allTransforms.star')
        #check if allTransforms exist
        if os.path.exists(transFormFile):
            print('Load distances from %s'%transFormFile)
            self.transList = tom_starread(transFormFile)
        else:
            self.transList = tom_calcTransforms(self.io['posAngList'], maxDistInPix, '',
                                                'exact', transFormFile, 1, worker_n)
            #self.transList should be a data frame object
    
    def groupTransForms(self, worker_n = 1, gpu_list = None, freeMem = None):
        '''
        give transformList,
        generate the clustering results(trees/classes of transforms) 
        maxChunk: the threshold to split the transformation data and 
                  parallel process
        '''
        if gpu_list is not None:
            worker_n = None
        print('Starting clustering');t1 = ti.default_timer()
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        outputFold = '%s/scores'%self.io['classifyFold']
        treeFile = '%s/tree.npy'%outputFold
        if os.path.exists(treeFile):
            print('loading tree')
            ll = np.load(treeFile) #load the clustering tree models 
        else:
            ll = tom_calcLinkage(self.transList, outputFold, maxDistInPix,
                                self.classify['cmb_metric'], worker_n, 
                                gpu_list,freeMem) #how to calculate the linkage 
        
        #ll should be a float32 n*4 ndarray, should consider GPU version
        clusters, _, _, thres= tom_dendrogram(ll, self.classify['clustThr'],self.transList.shape[0], 0, 0)
         
        self.classify['clustThr'] = thres  
        if len(clusters) == 0:
            print("Warninig: no classes! Check the threshold you input! This usually you put \
                  a very high or very low threshold.")
        else:
            #this step give the cluster id for each transform
            for single_dict in clusters:
                idx = single_dict['members']
                classes = single_dict['id']
                if len(idx) == 0: #no class detect 
                    continue
                colour = "%.2f-%.2f-%.2f"%(single_dict['color'][0],
                                           single_dict['color'][1],
                                           single_dict['color'][2])
                self.transList.loc[idx, "pairClass"] = classes
                self.transList.loc[idx, 'pairClassColour'] = colour
        print('Finishing clustering with %d seconds consumed'%(ti.default_timer()-t1))  
        
    def selectTransFormClasses(self, worker_n = 1, gpu_list = None, itrClean = 1):
        #now the translist has pairclass label as well as colour  label 
        '''
        select any class OR polysome and Relink
        itrClean: # of cycles to clean the data 
        '''
        transListSel = ''
        selFolds = ''
                
        if self.classify['relinkWithoutSmallClasses']:  
            for _ in range(itrClean):  
                #this step can keep classes we want for further analysis as well as remove class 0                 
                _,_, transListSelCmb = tom_selectTransFormClasses(self.transList,
                                                                  self.sel[0]['list'],
                                                                  self.sel[0]['minNumTransform'], '')
                self.transList = transListSelCmb
                #this select can discard the transforms with class ==0 (which failed to form cluster)
                os.rename('%s/scores/tree.npy'%self.io['classifyFold'],
                         '%s/scores/treeb4Relink.npy'%self.io['classifyFold'])
                self.groupTransForms(worker_n, gpu_list)
                
            if os.path.exists('%s/allTransforms.star'%self.io['classifyFold']):
                os.rename('%s/allTransforms.star'%self.io['classifyFold'],
                              '%s/allTransformsb4Relink.star'%self.io['classifyFold'])
            #store the translist
            header = { }
            header["is_loop"] = 1
            header["title"] = "data_"
            header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
            tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], self.transList,
                          header) 
                
        transListSel, selFolds, _ = tom_selectTransFormClasses(self.transList,
                                                               self.sel[0]['list'],
                                                               self.sel[0]['minNumTransform'],
                                                               '%s/classes'%(self.io['classifyFold'])) 
           
        if len(transListSel) == 0:
            print('Warning: no translist has been selected for further analysis!')
        return transListSel, selFolds  #transListSel be empty when on clustering performs  
    
    def alignTransforms(self):
        '''
        in each class, exchange the position of 
        ribosomes pairs to alignment
        '''
        warnings.filterwarnings('ignore')
        print('')
        print('Align the transform pairs')
        self.transList = tom_align_transformDirection(self.transList)
        #store the translist
        header = { }
        header["is_loop"] = 1
        header["title"] = "data_"
        header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
        tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], self.transList,
                      header)         
    
    def find_connectedTransforms(self, allClassesU = None, saveFlag = 1, 
                                 worker_n = 1, gpu_list = None):
        #the input should be the translist with classes information and direction aligned
        '''
        track the polyribosomes
        the branch analysis is developing
        '''
        warnings.filterwarnings('ignore')
        print('')
        print('Tracking the polysomes.')
        t1 = ti.default_timer()
        cmb_metric = self.classify['cmb_metric']
        pruneRad = self.transForm['maxDist']/self.transForm['pixS']
        allClasses = self.transList['pairClass'].values
        if allClassesU is None:
            allClassesU = np.unique(allClasses)
        
        allTomos = self.transList['pairTomoID'].values
        allTomosU = np.unique(allTomos)
        
        #if clean branch
        if self.transForm['branchDepth'] == 0:
            #load the information of avgshift/rot
            classSummaryList = '%s/stat/statPerClass.star'%self.io['classifyFold']
            if os.path.exists(classSummaryList):
                classSummaryList = tom_starread(classSummaryList)
            
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
                    avgShift = np.array([vectStat['meanTransVectX'], vectStat['meanTransVectY'], vectStat['meanTransVectZ']])
                    avgRot = np.array([angStat['meanTransAngPhi'], angStat['meanTransAngPsi'], angStat['meanTransAngTheta']])
                else:
                    avgShift = classSummaryList[classSummaryList['classNr'] == single_class].loc[:,
                                       ['meanTransVectX','meanTransVectY','meanTransVectZ']].values[0] #1D array
                    avgRot = classSummaryList[classSummaryList['classNr'] == single_class].loc[:,
                                       ['meanTransAngPhi','meanTransAngPsi','meanTransAngTheta']].values[0] #1D array
                                
                idx_drop = tom_find_poorBranch(self.transList.iloc[idx,:], avgShift, 
                                               avgRot, worker_n, gpu_list, cmb_metric, 
                                               pruneRad) #this function can find the index with branch, will be removed!
                if len(idx_drop) > 0:
                    print('find %d branches in classes:%d'%(len(idx_drop),single_class))
                    idx_rm = np.concatenate((idx_rm, idx_drop))
                
            if len(idx_rm) > 0:
                idxPair12_rm =  self.transList.loc[idx_rm, ['pairIDX1', 'pairIDX2']].values 
                self.transList = self.transList.drop(index = idx_rm)
                #update index 
                self.transList = self.transList.reset_index(drop=True)
                #update allClasses and allTomos
                allClasses = self.transList['pairClass'].values
                allClassesU_update = np.unique(allClasses)
                allClassesU = np.intersect1d(allClassesU_update, allClassesU, assume_unique=False)
                allTomos = self.transList['pairTomoID'].values
                allTomosU = np.unique(allTomos)
                
                
                    
        br = {} #1D-array, check the existence of branches
        for single_class in allClassesU:           
            if single_class == 0:  #also should aviod single_class == -1
                continue  ##no cluster occur with class == 0  
            if single_class == -1:
                print('Warning: track the polysomes w/o any cluster classes detected.\n  ==> Highly suggest do groupTransForms before this step')
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
        #only track the polysomes with class > 0 as well as in each tomo
        #those with class 0 will have pairLabel == -1 
        #this step is clean the transform which has pairLabel == 0
        len_branch = [key for key in br.keys() if br[key] > 0]
        if len(len_branch) > 0:
            print('Warning: found branches in these class: %s'%(str(len_branch)))
            print('==> make smaller classes')
       
        print('Polysome tracking finished with %.5f seconds consumed'%(ti.default_timer() - t1))
       
        if saveFlag:
            header = { }
            header["is_loop"] = 1
            header["title"] = "data_"
            header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
            tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], self.transList, header)
            
        if self.transForm['branchDepth'] == 0:
            if len(idx_rm) >0:
                return idxPair12_rm
        else:
            return None
        
    def analyseTransFromPopulation(self,  outputFolder = '', visFolder = '', classNr = -1, verbose = 1):
        '''
        this method gives a summary of the polysomes track in each 
        transforms class, like if has branch/the length of the polysome/the order, 
        all is saved at the stat folder with starfiles
        '''
        print('')
        print('Summary polysomes')
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
        writeOutputStar(stat, statPerPoly, outputFolder)
        genOutput(stat, minTransMembers = 10)
    
    def find_transFromNeighbours(self, outputName = '', nrStatOut = 10):
        '''
        for each ribosome of one ribosome pair, this ribosome can belong to 
        different transform clusters, that is to say, can be in different ribosome 
        pairs and in different cluster classes.
        This function summary the cluster classes for each ribosome

        '''
        if len(outputName) == 0:
            outputName = "%s/allTransforms.star"%self.io['classifyFold']
        allTomoId = self.transList['pairTomoID'].values
        allTomoIdU = np.unique(allTomoId)
        
        listTomo = [ ]
        idx = [ ]
        neighNpT = [ ]
        neighNMT = [ ]
        
        for i in range(len(allTomoIdU)):
            idx.append(np.where(allTomoId == allTomoIdU[i])[0])
            listTomo.append(self.transList.iloc[idx[i], :])
        
        for i in range(len(allTomoIdU)):
            listTomo[i], single_neighNpT, single_neightNMT  = findNeighOneTomo(listTomo[i])
            neighNpT.append(single_neighNpT)
            neighNMT.append(single_neightNMT)
        
        neigh_N_plus = np.array([], dtype = np.int).reshape(-1,3)
        neigh_N_minus = np.array([], dtype = np.int).reshape(-1,3)
        
        for i in range(len(listTomo)):
            self.transList.loc[idx[i], :] = listTomo[i]
            neigh_N_plus = np.concatenate((neigh_N_plus,neighNpT[i]), axis = 0 )
            neigh_N_minus = np.concatenate((neigh_N_minus, neighNMT[i]), axis = 0)
            
        nCmb = np.concatenate((neigh_N_plus, neigh_N_minus), axis = 1)
        #discard the row with [0,0,0,0,0,0]
        nCmb = nCmb[~np.all(nCmb == 0, axis =1 )]
        nCmbU, ic = np.unique(nCmb, return_inverse = True, axis = 0)
        #print(nCmbU)
        clCount = Counter(ic)
        key_list = [ ]
        value_list =[ ]
        for key in clCount:
            key_list.append(key)
            value_list.append(clCount[key])
        key_array = np.array(key_list, dtype = np.int)
        value_array = np.array(value_list, dtype = np.int)
        
        #plot the results
        plt.figure()
        plt.bar(key_array,  value_array)
        plt.show()
        plt.close()
        
        clIdx_Sort = np.argsort(value_array)[::-1]
        clCount_Sort = value_array[clIdx_Sort]
        clRow_Sort = key_array[clIdx_Sort]
        
        
        if nrStatOut > len(clCount_Sort):
            nrStatOut == len(clCount_Sort)
        print('classn+1        ||    classn-1    abundance')
        for i in range(nrStatOut):
            print("  % d %d %d\t||\t%d %d %d\t\t%.2f"%(nCmbU[clRow_Sort[i],0], 
                                                    nCmbU[clRow_Sort[i], 1],
                                                    nCmbU[clRow_Sort[i],2],
                                                    nCmbU[clRow_Sort[i], 3],
                                                    nCmbU[clRow_Sort[i],4],
                                                    nCmbU[clRow_Sort[i],5],
                                                     clCount_Sort[i]/self.transList.shape[0]*100) + "%")
        if len(outputName) != 0:
            header = { }
            header["is_loop"] = 1
            header["title"] = "data_"
            header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
            tom_starwrite(outputName, self.transList, header)
          
    def analyseConfClasses(self):  
        '''
        find the pattern of classes of ribosomes 
        in each polysome (creating)
        '''
        if self.clSt['findPat']['classNr'] == -2:
            print('Skipping conf. Class pattern analysis ==> no conf class selected')
            return 
        inputFile = '%s/stat/statPerPoly.star'%self.io['classifyFold']
        outputFold = '%s/stat/confAnalysis'%self.io['classifyFold']
        if not os.path.exists(outputFold):
            os.mkdir(outputFold)
        tom_findPattern(inputFile, 99.0, 1000, outputFold, self.clSt['findPat'])
        
      
    def genOutputList(self, transListSel, outputFoldSel):
        '''
        output the summary of polysomes of each transform class.
        include each tomogram and the whole tomogram for one transform
        class
        ATTENTION: the transListSel in this script is without any polysome ID info!!
        '''
        print('Generating selection lists and summary for each polysome.')
         
        for i in range(len(transListSel)):  #each translist represents one translist of one class
            transListTmp = transListSel[i]
            outputFoldCenter = "%s/pairCenter/"%outputFoldSel[i]
            tom_genListFromTransForm(transListTmp, outputFoldCenter, 'center')
            outputFoldCenter = "%s/particleCenter/"%outputFoldSel[i]
            tom_genListFromTransForm(transListTmp, outputFoldCenter, 'particle')

        
    def generateTrClassAverages(self):
        '''
        using relion/chimera to average the ribosome from one transform class
        '''
        if np.isinf(self.avg['filt']['minNumTransform']):
            print('Skipping translational class averaging density map')
            return 
        wk = "%s/classes/c*/particleCenter/allParticles.star"%self.io['classifyFold']
        outfold = '%s/avg/exp/%s'%(self.io['classifyFold'], self.io['classificationRun'])
        outfoldVis = '%s/vis/averages'%self.io['classifyFold']
        tom_avgFromTransForm(wk, self.avg['filt'], self.avg['maxRes'], 
                             self.avg['pixS'],outfold, self.avg['command'])
        
        classFile = "%s/stat/statPerClass.star"%self.io['classifyFold']
        scriptName = "%s/avg.cmd"%(outfoldVis)
        
        genMapVisScript(outfold, classFile, scriptName, self.transList, self.transForm['pixS'], 0)
    
    def genTrClassForwardModels(self):
        '''
        give the transvector and transangles, 
        this function can generate the forward model \
        polysomes (one by one)
        '''
        if not os.path.exists(self.fw['Map']):
            print('Skipping forward model generation')
            return 
        
        outfold = '%s/avg/model/%s/c'%(self.io['classifyFold'], 
                                       self.io['classificationRun'])
        inputList = '%s/stat/statPerClass.star'%self.io['classifyFold']
        
        scale4shift = self.transForm['pixS']/self.fw['pixS']
#       tom_genForwardMapPairTransForm(inputList, self.fw['Map'], self.fw['numForwardRepeats'],
#                                       -1, outfold, scale4shift, self.fw['minNumTransform'])
        
        outfoldVis = '%s/vis/averages'%(self.io['classifyFold'])
        #classFile = '%s/stat/statPerClass.star'%self.io['classifyFold']
        scriptName = '%s/models.cmd'%outfoldVis
        genMapVisScript(outfold, inputList, scriptName, self.transList, self.transForm['pixS'], 0 )
        
    def visResult(self):
        '''
        visulize the polysomes 
        as well as the linkage results
        '''
        
        print(' ')
        print('Rendering figures')
        vectVisP = self.vis['vectField']
        if vectVisP['render']:
            tom_plot_vectorField(self.transList, vectVisP['showTomo'], 
                                  vectVisP['showClass'], vectVisP['polyNr'], vectVisP['onlySelected'],
                                  vectVisP['repVectLen'],vectVisP['repVect'],
                                  np.array([0,0,1]), '', '%s/vis/vectfields'%self.io['classifyFold'])
        else:
            print('VectorFiled rendering skipped')
               
        treeFile = '%s/scores/tree.npy'%self.io['classifyFold']
        thres = self.classify['clustThr']
        
        for i in [1,2]:
            if i==1:
                self.dspTree(self.io['classifyFold'], treeFile, self.classify['clustThr'], -1)
            if i==2:
                self.dspLinkage(self.io['classifyFold'], treeFile, thres)
        print('Rendering figures done')
    
    @staticmethod
    def dspTree(classifyFold, treeFile, clustThr, nrTrans=-1):
        print(' ')
              
        _, _, _, _ = tom_dendrogram(treeFile, clustThr, nrTrans, 1, 500)
        
        plt.ylabel('linkage score')        
        plt.savefig('%s/vis/clustering/tree.png'%classifyFold, dpi = 300)
        plt.show()
        plt.close()
        
    @staticmethod
    def dspLinkage(classifyFold, treeFile,thres):
        plt.figure()
        plt.title('link-levels')
        tree = np.load(treeFile)
        plt.plot(np.sort(tree[:,2])[::-1], label = 'link-level')
        plt.plot(np.sort(tree[:,2])[::-1], 'ro', label = 'link-level')
        plt.plot(
                np.ones(tree.shape[0])*thres, 'k--' ,label = 'threshold')
        plt.legend()
        plt.text(1,thres*2, 'threshold = %.2f'%thres)
        plt.xlabel('# of transforms pairs')
        plt.ylabel('linkage score')
        
        plt.savefig('%s/vis/clustering/linkLevel.png'%classifyFold, dpi = 300)
        plt.show()
        plt.close()
        
    def link_ShortPoly(self):
        '''
        link shorter polysomes 
        logic: put one ribo at end of each poly, and judge if this ribo 
        can link head ribosome of other polysomes from the same trans class
        '''
        if self.fillPoly['addNum'] == 0:
            print('skiping filled up ribosomes')
            return
        
        if self.fillPoly['class'][0] < 0:
            print('')
            print('Fill up polysome in all cluster classes')
            classList = np.unique(self.transList['pairClass'].values)
        else:
            print('Fill up these polysome classes %s.'%str(self.fillPoly['class']))
            classList = self.fillPoly['class']
        #load summary star file
        classSummaryList = '%s/stat/statPerClass.star'%self.io['classifyFold']
        if os.path.exists(classSummaryList):
            classSummary = tom_starread(classSummaryList)
        else:
            print('lacking file:%s, please run analyseTransFromPopulation!'%classSummaryList)   
            return
        #using networkx(tom_connectGraph) to find the ribosomes to link OR to be linked
        statePolyAll_forFillUp = tom_connectGraph(self.transList)
        (filepath,tempfilename) = os.path.split(self.io['posAngList'])
        (shotname,extension) = os.path.splitext(tempfilename)           
        #give the store name & path of the particle star and transList
        transListOutput = '%s/allTransformsFillUp.star'%self.io['classifyFold']
        particleOutput = '%s/%sFillUp.star'%(filepath,shotname)
        
        for pairClass in classList:
            if pairClass == -1:
                print('Warning: can not detect transformation classes!')
                return
            if pairClass == 0:
                continue        
            #get the information of ribos to link OR to be linked 
            statePolyAll_forFillUpSingleClass = statePolyAll_forFillUp[statePolyAll_forFillUp['pairClass'] == pairClass ]
            #load the fit summary data 
            fitData = '%s/vis/fitDist/distFit_c%d.csv'%(self.io['classifyFold'], pairClass)
            if os.path.exists(fitData):
                fitData = pd.read_csv(fitData,sep = ",")
            #read the avgshift and avgRot 
            avgShift = classSummary[classSummary['classNr'] == pairClass].loc[:,
                                   ['meanTransVectX','meanTransVectY','meanTransVectZ']].values[0] #1D array
            avgRot = classSummary[classSummary['classNr'] == pairClass].loc[:,
                                   ['meanTransAngPhi','meanTransAngPsi','meanTransAngTheta']].values[0] #1D array
                       
            #automatically choose the method 1.extreme ,2.lognorm           
            if isinstance(fitData,str):
                method = 'extreme'
                cmbDistMax = classSummary[classSummary['classNr'] == pairClass]['maxCNDist'].values[0]                             
                cmbDistMeanStd = (0,0)              
                fitParam = ''
            else:
                method = 'lognorm'
                cmbDistMax =  0   
                cmbDistMeanStd = (classSummary[classSummary['classNr'] == pairClass]['meanCNDist'].values[0],
                                classSummary[classSummary['classNr'] == pairClass]['stdCNDist'].values[0])
                fitParam = fitData['fit_params'][0]
            print('Linking polys of class:%d using %s'%(pairClass, method))  
    
            self.transList = tom_addTailRibo(statePolyAll_forFillUpSingleClass, self.transList, pairClass, avgRot, 
                                             avgShift,cmbDistMax,cmbDistMeanStd,fitParam,
                                             self.io['posAngList'], 
                                             self.transForm['maxDist']/self.transForm['pixS'],                                         
                                             transListOutput,  particleOutput,
                                             self.fillPoly['addNum'], self.fillPoly['riboinfo'], method)                

        #retrack the polysomes 
        self.transList['pairLabel'] = -1
        self.transList['pairPosInPoly1'] = -1
        self.transList['pairPosInPoly2'] = -1
        self.transForm['branchDepth'] = 0 #clean the branches
        self.find_connectedTransforms(classList, 0)
        #save the transList 
        saveStruct(transListOutput,self.transList)
        
    def visPoly(self, lenPoly = 10):
        '''
        this method is aimed to vis all polysomes with length longer than 5
        '''
        #keep_idx = [ ]
        
        transList  = self.transList
        classU = np.unique(transList['pairClass'])
        for singleClass in classU:
            if singleClass == -1:
                print('No trans classes detected!')
                return
            if singleClass == 0:
                continue
            
            transListClass = transList[transList['pairClass'] == singleClass]
            polyLenList = transListClass['pairLabel'].value_counts()
            polyLenBig = np.asarray(polyLenList[polyLenList>lenPoly].index)
            if len(polyLenBig) == 0:
                continue
#            keep_row = np.where(transListClass['pairLabel'].values == polyLenBig[:,None])[-1]
#            keepIdxSingle = list(transListClass.index[keep_row])
            for singlePoly in polyLenBig:
                print('Class', singleClass, ':', singlePoly)
                keep_row = np.where(transListClass['pairLabel'].values == singlePoly)[0]                
                transListPlot = transListClass.iloc[keep_row,:]
                tom_plot_vectorField(transListPlot)
                del transListPlot
            del transListClass          
            #keep_idx.extend(keepIdxSingle)
#        if len(keep_idx) == 0:
#            print('It seems that no polysome longer than %d'%lenPoly)
#            return
#        transList = transList.loc[keep_idx,:]
#        tom_plot_vectorField(transList)
        
        