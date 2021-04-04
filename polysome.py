import os
import numpy as np
import shutil

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
        self.sel = [ ]
        self.sel[0] = { }
        self.sel[0]['classNr'] = -1
        self.sel[0]['polyNr'] = -1
        self.sel[0]['list'] = 'Classes-Sep'
        #for vis
        self.vis  = { }
        self.vis["vectField"] = { }
        self.vis['vectField']['render'] = 1
        self.vis['vectField']['showTomo'] = -1
        self.vis['vectField']['showClass'] = np.arange(10000) #10^4 classes?!
        self.vis['vectField']['onlySelected'] = 1
        self.vis['vectField']['polyNr'] = -1
        self.vis['vectFiled']['repVect'] = np.array([0,1,0])
        self.vis['vectFiled']['repVectLen'] = 20
        #for avg
        self.avg = { }
        self.avg['command'] = 'tom'
        self.avg['filt'] = { }
        self.avg['filt']['minNumTransform'] = 0
        self.avg['filt']['maxNumPart'] = 500
        self.avg['pixS'] = 3.42
        #for fw
        self.fw = { } 
        self.fw['Map'] = 'vol4forward.mrc'
        self.fw['minNumTransform'] = 0
        self.fw['pixS'] = self.transForm['pixS']
        #for Conf class Analysis
        self.clSt = { }
        self.clSt['findPat'] = { }
        self.clSt['findPat']['classNr'] = -2
        self.clSt['findPat']['filt'] = { }
        self.clSt['findPat']['filt']['operator'] = '>'
        self.clSt['findPat']['filt']['value'] = 3
        #for the trand data 
        self.transList = translist
        
    def creatOutputFolder(self):
        self.io['classifyFold'] = '%s/%s'%(self.io['projectFolder'],
                                           self.io['classificationRun'])
        projectFolder = self.io['projectFolder']
        classificationFolder = self.io['classifyFold']
        if not os.path.exists(projectFolder):
            os.mkdir(projectFolder)
        if not os.path.exists(classificationFolder):
            os.mkdir(classificationFolder)
            os.mkdir('%s/scores'%classificationFolder)
            os.mkdir('%s/vis'%classificationFolder)
            os.mkdir('%s/vis/vectfileds'%classificationFolder)
            os.mkdir('%s/vis/clustering'%classificationFolder)
            os.mkdir('%s/vis/averages'%classificationFolder)
            os.mkdir('%s/stat'%classificationFolder)
            os.mkdir('%s/avg'%classificationFolder)
            os.mkdir('%s/avg/exp'%classificationFolder)
            os.mkdir('%s/avg/model'%classificationFolder)
        #remove the dirs
        if  os.path.exists(classificationFolder):
            if os.path.isdir('%s/stat'%classificationFolder):
                shutil.rmtree('%s/stat'%classificationFolder)
            if os.path.isdir('%s/classes'%classificationFolder):
                shutil.rmtree('%s/classes'%classificationFolder)        
            if os.path.isdir('%s/avg'%classificationFolder):
                shutil.rmtree('%s/avg'%classificationFolder)         
            if os.path.isdir('%s/vis'%classificationFolder):
                shutil.rmtree('%s/vis'%classificationFolder)   
    def calcTransForms(self):
        
        
        
         
        
        
        
        
    
    
