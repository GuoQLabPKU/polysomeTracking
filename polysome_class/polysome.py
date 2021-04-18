import os
import numpy as np
import shutil
from scipy.cluster.hierarchy import dendrogram
import warnings
from collections import Counter
import matplotlib.pyplot as plt

from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from py_transform.tom_calcTransforms import tom_calcTransforms
from py_cluster.tom_calcLinkage import tom_calcLinkage
from py_cluster.tom_dendrogram import tom_dendrogram
from py_cluster.tom_selectTransFormClasses import tom_selectTransFormClasses
from py_align.tom_align_transformDirection import tom_align_transformDirection
from py_link.tom_linkTransforms import tom_linkTransforms
from py_summary.tom_analysePolysomePopulation import *
from py_link.tom_find_transFormNeighbours import *
from py_summary.tom_findPattern import tom_findPattern
from py_summary.tom_genListFromTransForm import tom_genListFromTransForm
from py_summary.tom_avgFromTransForm import tom_avgFromTransForm
from py_summary.genMapVisScript import genMapVisScript
from py_vis.tom_plot_vectorField import tom_plot_vectorField


class Polysome:
    ''' 
    Polysome is one class that used to track the polysomes in
    the risome groups
    '''
    def __init__(self, input_star = None, run_time = 'run0',translist = None):
        '''
        set the default properties for polysome class
        input:
            the star file for ribosomes euler angles and coordinates
            the times to get new polysomes
        '''
        #for io
        self.io = { }
        self.io['posAngList'] = input_star
        (filepath,tempfilename) = os.path.split(self.io['posAngList']);
        (shotname,extension) = os.path.splitext(tempfilename)
        self.io['projectFolder'] = 'cluster-%s'%shotname
        self.io['classificationRun'] = run_time 
        self.io['classifyFold'] = None  #should be modify only after run input
        #for trans
        self.transForm = { }
        self.transForm['pixS'] = 3.42
        self.transForm['maxDist'] = 342
        self.transForm['branchDepth'] = 2
        #for classify
        self.classify = { }
        self.classify['clustThr'] = 0.12
        self.classify['relinkWithoutSmallClasses'] = 1
        self.classify['cmb_metric'] = 'scale2Ang'
        #for select the clean polysomes
        self.sel = [ ]  #shoule be a list
        self.sel.append({})
        self.sel[0]['classNr'] = np.array([-1]) #select the class of transform
        self.sel[0]['polyNr'] = np.array([-1])   #select the polysome
        self.sel[0]['list'] = 'Classes-Sep'
        #for vis
        self.vis  = { }
        self.vis["vectField"] = { }
        self.vis['vectField']['render'] = 1
        self.vis['vectField']['showTomo'] = np.array([-1])  #what tomos to vis?
        self.vis['vectField']['showClass'] = np.arange(10000) #what classes of trans to vis?
        self.vis['vectField']['onlySelected'] = 1  #only vis thoes selected by upper to params?
        self.vis['vectField']['polyNr'] = np.array([-1]) #what polys to show?
        self.vis['vectField']['repVect'] = np.array([[0,1,0]]) #how to represents the ribosomes(angle + position)
                                                                #must be 2D array
        self.vis['vectField']['repVectLen'] = 20 
        #for avg
        self.avg = { }
        self.avg['command'] = 'tom'
        self.avg['filt'] = { }
        self.avg['filt']['minNumTransform'] = np.inf   #to select the particles in each class
        self.avg['filt']['maxNumPart'] = 500  #select number of transforms in each class
        self.avg['pixS'] = 3.42
        self.avg['maxRes'] = 45
        #for fw
        self.fw = { } 
        self.fw['Map'] = 'vol4forward.mrc'
        self.fw['minNumTransform'] = 0
        self.fw['pixS'] = self.transForm['pixS']
        #for Conf class Analysis
        self.clSt = { }
        self.clSt['findPat'] = { }
        self.clSt['findPat']['classNr'] =  np.array([-2])  #try to make this an array #-2
        self.clSt['findPat']['filt'] = { }
        self.clSt['findPat']['filt']['operator'] = '>'
        self.clSt['findPat']['filt']['value'] = 3
        #for the trand data 
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
        if  os.path.exists(classificationFolder): #it seems that only scores dir kept
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
        os.mkdir('%s/stat'%classificationFolder)
        os.mkdir('%s/avg'%classificationFolder)
        os.mkdir('%s/avg/exp'%classificationFolder)
        os.mkdir('%s/avg/model'%classificationFolder)
        os.mkdir('%s/classes'%classificationFolder)
            
            
    def calcTransForms(self):
        '''
        give the input ribosomes input star,
        generate the transforms 
        '''
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        transFormFile = '%s/%s'%(self.io['classifyFold'],
                                 'allTransforms.star')
        #check if allTransforms exist
        if os.path.exists(transFormFile):
            print('load distances from %s'%transFormFile)
            self.transList = tom_starread(transFormFile)
        else:
            self.transList = tom_calcTransforms(self.io['posAngList'], maxDistInPix, '',
                                                'exact', transFormFile)
            #self.transList should be data frame object
    
    def groupTransForms(self):
        '''
        give transforms,
        generate the clustering results(trees/classes) 
        '''
        maxDistInPix = self.transForm['maxDist']/self.transForm['pixS']
        outputFold = '%s/scores'%self.io['classifyFold']
        treeFile = '%s/tree.npy'%outputFold
        if os.path.exists(treeFile):
            ll = np.load(treeFile) #load the clustering tree models 
        else:
            ll = tom_calcLinkage(self.transList, outputFold, maxDistInPix,
                                self.classify['cmb_metric'] ) #how to calculate the linkage 
        #ll should be a float64 n*4 ndarray
        clusters, _, _, thres = tom_dendrogram(ll, 
                                                 self.classify['clustThr'],
                                                 self.transList.shape[0], 0, 500)

        #groups shoud be a list with dicts stored
        #threshold is to kept transforms with distance less than this threshold
        #large distance represents less similariity to form new cluster            
        self.classify['clustThr'] = thres  #this is differnt from matlab clusterThrAct
        if len(clusters) == 0:
            print("Warninig: no classed! Check the threshold you input!")
            dendrogram(ll) #using default parameters to show the tree
        else:
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
                    
    def selectTransFormClasses(self):
        #now the translist has pairclass label as well as colour  label 
        '''
        select any class OR polysome  and Relink
        '''
        transListSel = ''
        selFolds = ''
        
        itrClean = 1
        for _ in range(itrClean):
            if self.classify['relinkWithoutSmallClasses']:          
                _,_, transListSelCmb = tom_selectTransFormClasses(self.transList,
                                                                  self.sel[0]['list'],
                                                                  '', self.sel[0]['minNumTransform'])
                self.transList = transListSelCmb
                #this select can discard the transforms with class ==0 (which failed to form cluster)
                os.rename('%s/scores/tree.npy'%self.io['classifyFold'],
                         '%s/scores/treeb4Relink.npy'%self.io['classifyFold'] )
                self.groupTransForms()
                if os.path.exists('%s/allTransforms.star'%self.io['classifyFold']):
                    os.rename('%s/allTransforms.star'%self.io['classifyFold'],
                              '%s/allTransformsb4Relink.star'%self.io['classifyFold'])
                #store the translist
                header = { }
                header["is_loop"] = 1
                header["title"] = "data_"
                header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
                tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], self.transList)
                
            transListSel, selFolds, _ = tom_selectTransFormClasses(self.transList,
                                       self.sel[0]['list'],
                                        self.sel[0]['minNumTransform'], #just remove small classes[0] w/o relink
                                       '%s/classes'%(self.io['classifyFold'])) 

           #the transListSel & selFolds can be [] with empty or with elements            
        return transListSel, selFolds  #transListSel be empty when on clustering performs  
    
    def alignTransforms(self):
        '''
        in each class, exchange the position of 
        ribosomes pairs to alignment
        '''
        print('Align the transform pairs')
        tom_align_transformDirection(self.transList)
    
    def find_connectedTransforms(self):
        #the input should be the translist with classes confered and directon aligned
        '''
        track the polyribosomes
        '''
        warnings.filterwarnings('ignore')
        print('tracking the polysomes.')
        allClasses = self.transList['pairClass']
        allClassesU = np.unique(allClasses)
        
        allTomos = self.transList['pairTomoID']
        allTomosU = np.unique(allTomos)
        
        br = np.zeros(len(allClassesU), dtype = np.int) #1D-array
        
        for single_class in allClassesU:
            if single_class == 0:  #also should aviod single_class == -1
                continue  ##no cluster occur with class == 0  
            if single_class == -1:
                print('Warning: track the polysomes w/o any classes of transforms.')
                
            idx1 = np.where(allClasses == single_class)[0]
            offset_PolyID = 0
            for single_tomo in allTomosU:
                idx2 = np.where(allTomos == single_tomo)[0]
                idx = np.intersect1d(idx1, idx2)
                #track the single polysome in the same tomogram 
                if len(idx) > 1:
                    self.transList.loc[idx,:], _, br[single_class], offset_PolyID = tom_linkTransforms(self.transList.loc[idx,:],
                                      self.transForm['branchDepth'], offset_PolyID)
                elif len(idx) == 1:
                    offset_PolyID += 1
                    self.transList.loc[idx,'pairLabel'] = offset_PolyID  #begin with 1
                    self.transList.loc[idx,'pairPosInPoly1'] = 1 #the order in each polysome, begin with 1
                    self.transList.loc[idx,'pairPosInPoly2'] = 2 #the order in branch, ~= poly1 +1 
                    
        
        #only track the polysomes with class > 0 as well as in each tomo
        #those with class 0 will have pairLabel == -1
        len_branch = np.where(br > 0)[0]
        if len(len_branch) > 0:
            print('Warning: found brancges in class %s'%(str(len_branch)))
            print('==> make smaller classes')
        
        header = { }
        header["is_loop"] = 1
        header["title"] = "data_"
        header["fieldNames"]  = ["_%s"%i for i in self.transList.columns]
        tom_starwrite('%s/allTransforms.star'%self.io['classifyFold'], self.transList, header)
        
    def analyseTransFromPopulation(self,  outputFolder = '', classNr = -1, repVolume = '', verbose = 1):
        '''
        this method gives a summary of the polysomes track in each 
        transforms class, like if has branch/the length of the polysome/the order, 
        all is saved at the stat folder with starfiles
        '''
        if outputFolder == '':
            outputFolder = '%s/stat'%self.io['classifyFold']
        allClasses = self.transList['pairClass']
        allClassesU = np.unique(allClasses)
        stat = [ ]
        statPerPolyTmp = [ ]
        for single_class in allClassesU:
            idx = np.where(allClasses == single_class)[0]
            stat.append(analysePopulation(self.transList.iloc[idx,:]))
            statPerPolyTmp.append(analysePopulationPerPoly( self.transList.iloc[idx,:]))
        
        statPerPoly = sortStatPoly(statPerPolyTmp)
        stat = sortStat(stat)
        writeOutputStar(stat, statPerPoly, outputFolder)
        genOutput(stat,2)
    
    def find_transFromNeighbours(self, outputName = '', nrStatOut = 10):
        '''
        analyse the transform classes for each ribosomes(this ribosome should be in different 
        transform classes)
        '''
        if len(outputName) == 0:
            outputName = "%s/allTransforms.star"%self.io['classifyFold']
        allTomoId = self.transList['pairTomoID']
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
        
        #plot
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
            print('skipping conf. Class analysis ==> no conf class selected')
            return 
        inputFile = '%s/stat/statPerPoly.star'%self.io['classifyFold']
        outputFold = '%s/stat/confAnalysis'%self.io['classifyFold']
        if  not os.path.exists(outputFold):
            os.mkdir(outputFold)
        tom_findPattern(inputFile, 99.0, 1000, outputFold, self.clSt['findPat'])
        
      
    def genOutputList(self, transListSel, outputFoldSel):
        '''
        output the summary of polysomes of each transform class.
        include each tomogram and the whole tomogram for one transform
        class
        '''
        print('generating selection lists')
         
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
            print('skipping translational class averaging')
            return 
        wk = "%s/classes/c*/particleCenter/allPart.star"%self.io['classifyFold']
        outfold = '%s/avg/exp/%s'%(self.io['classifyFold'], self.io['classificationRun'])
        outfoldVis = '%s/vis/averages'%self.io['classifyFold']
        tom_avgFromTransForm(wk, self.avg['filt'], self.avg['maxRes'], 
                             self.avg['pixS'],outfold,  self.avg['command'])
        
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
            print('skipping forward model generation')
            return 
        
        outfold = '%s/avg/model/%s/c'%(self.io['classifyFold'], 
                                       self.io['classificationRun'])
        inputList = '%s/stat/statPerClass.star'%self.io['classifyFold']
        
        scale4shift = self.transForm['pixS']/self.fw['pixS']
#        tom_genForwardMapPairTransForm(inputList, self.fw['Map'], self.fw['numForwardRepeats'],
#                                       -1, outfold, scale4shift, self.fw['minNumTransform'])
        
        outfoldVis = '%s/vis/averages'%(self.io['classifyFold'])
        #classFile = '%s/stat/statPerClass.star'%self.io['classifyFold']
        scriptName = '%s/models.cmd'%outfoldVis
        genMapVisScript(outfold, inputList, scriptName, self.transList, self.transForm['pixS'],0 )
        
    def visResult(self):
        print(' ')
        print('rendering figures')
        vectVisP = self.vis['vectField']
        if vectVisP['render']:
            tom_plot_vectorField(self.transList, vectVisP['showTomo'], 
                                  vectVisP['showClass'], vectVisP['polyNr'], vectVisP['onlySelected'],
                                  vectVisP['repVectLen'],vectVisP['repVect'],
                                  np.array([0,0,1]), '', '%s/vis/vectfields'%self.io['classifyFold'])
        else:
            print('vectorFiled rendering skipped')
        
        nrTrans = self.transList.shape[0]
        treeFile = '%s/scores/tree.npy'%self.io['classifyFold']
        thres = self.classify['clustThr']
        
        for i in [1,2]:
            if i==1:
                self.dspTree(self.io['classifyFold'], self.classify['clustThr'], nrTrans, treeFile)
            if i==2:
                self.dspLinkage(self.io['classifyFold'], treeFile, thres)
        print('rendering figures done')
    
    @staticmethod
    def dspTree(classifyFold, clustThr, nrTrans, treeFile):
        print(' ')
              
        _, _, _, _,_  = tom_dendrogram(treeFile, clustThr, nrTrans )
        
        plt.ylabel('linkage score')        
        plt.savefig('%s/vis/clustering/tree.png'%classifyFold, dpi = 300)
        plt.show()
        #plt.close()
        
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
        #plt.close()
        
         
        
         
         
        
        
        
        
        
        
        
        
            
        
        
        
        
    
    
        
            
            
        
        
        
        
                    
                
        
    
    
        
        
        

            
    
    
    
            
        
        
        
        
        
    
            
            
        
            
            
            
            
            
            
            
        
        
        
        
        
                        
                        
                
            
            
            
            
        

            
        
            
            
            
        
        
        
        
        
        
         
        
        
        
        
    
    
