#import pandas as pd
import numpy as np

from nemotoc.py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from nemotoc.py_io.tom_starread import tom_starread

def tom_extractData(listFile, pixS = -1, makePosUnique = 0):
    '''
    tom_extractData reads positions,angles... from file after tom_starread (.star)
  
    angles,positions,shifts,cmbInd=tom_extractData(listFile)
  
    PARAMETERS
  
    INPUT
    
        listFile                  dataframe from tom_starread(starfile)['data_particles'] OR starfile name
        makePosUnique             (0) flag for removing duplicates

    OUTPUT
  
        st                        dict containing list information 
      
        p1.angles                 angles (nx3) in zxz
        p1.positions              positions  (nx3)          
        p1.shifts                 shifts (nx3)
        p1.classes                classes 
        p1.tomoNames              tomoName (n) cell
        p1.tomoIDs                tomoNames translated inot ids 
        p1.psfs                   point spread for particles
        p1.pixs                   pixelsize in Ang
        p1.cmbInd                 cmbInd index of  pairs only for pair files 


    EXAMPLE
       starfile = tom_starread("in.star")
       st = tom_extractData(starfile);

    REFERENCES
  
    NOTE:
    
    '''
    supportType = ['DataFrame', 'dict', 'str']
    type_inputfile = type(listFile).__name__
    if type_inputfile not in supportType:
        raise TypeError('''You should input the dataframe/dict using tom_starread!
                        Or starfile name!''')
    else:
        data_process, type_processData, pixelSize = readList(listFile)
        if pixS > 0:
            pixelSize = pixS           
        if type_processData == 'pairStar':
            st = extractFromPairStar(data_process, pixelSize)

        if type_processData == "relion2Star":
            st = extractFromRelionStar2(data_process, pixelSize)

        if type_processData == "relion3Star":
            st = extractFromRelionStar3(data_process, pixelSize)
           
        if type_processData == 'stopGapStar':
            st = extractFromStopGapStar(data_process, pixelSize)

    if 'tomoID' not in st["label"].keys():  #update the tomoID,such that number replace tomo Name
        st = updateTomoID(st)        
    if st["label"]["tomoID"][0] == -1:
        st = updateTomoID(st)
    return st
        
def readList(listFile): 
    type_inputfile = type(listFile)
    if isinstance(listFile, str):
        if '.star' in listFile:
            list_return = tom_starread(listFile)
            list_returnData = list_return['data_particles']
            pixelSize = list_return['pixelSize']
            
            if 'pairTransVectX' in list_returnData.columns:
                return list_returnData, 'pairStar', pixelSize 
            
            elif 'rlnCoordinateX' in list_returnData.columns:
                if 'rlnOpticsGroup' in list_returnData.columns:
                    return list_returnData, 'relion3Star', pixelSize
                else:
                    return list_returnData, 'relion2Star', pixelSize
                
            elif 'phi' in list_returnData.columns:
                return list_returnData, 'stopGapStar', pixelSize
            
        else:
            raise TypeError("Unrecognized input type!")
            
            
    elif type_inputfile.__name__ == "DataFrame":
        list_return = listFile
        if 'pairTransVectX' in list_return.columns:   
            return list_return, 'pairStar', listFile['pairPixelSizeAng'].values[0]
        
        elif 'rlnCoordinateX' in list_return.columns:
            if 'rlnOpticsGroup' in list_return.columns:
                return list_return, 'relion3Star', -1,
            else:
                return list_return, 'relion2Star', listFile['rlnDetectorPixelSize'].values[0]
                               
        elif 'phi' in list_return.columns:
            return list_return, 'stopGapStar', -1
        
        else:
            raise TypeError("Unrecognized input type!")
    
    elif isinstance(listFile, dict):
        if listFile['type'] == 'relion2':
            list_return = listFile['data_particles']
            if 'pairTransVectX' in list_return.columns:
                return list_return, 'pairStar', listFile['pixelSize']
            elif 'rlnCoordinateX' in list_return.columns:
                return list_return, 'relion2Star', listFile['pixelSize']
            else:
                raise TypeError("Unrecognized input type!")
                
        elif listFile['type'] == 'relion3':
            return listFile['data_particles'], 'relion3Star', listFile['pixelSize']
         
        elif listFile['type'] == 'stopgap':
            return listFile['data_particles'], 'stopGapStar', listFile['pixelSize']
        
        else:
            raise TypeError("Unrecognized input type!")

    else:
        raise TypeError("Unrecognized input type!")
        
            
def extractFromPairStar(starfile, pixelSize):
    cmbInd = np.zeros([starfile.shape[0],7],dtype = np.int)
    data_dict  = { }
    data_dict["p1"] = { }
    data_dict["p2"] = { }
    data_dict["label"] = { }
    data_dict["p1"]["positions"] = np.array(starfile.loc[:,["pairCoordinateX1","pairCoordinateY1","pairCoordinateZ1"]])
    data_dict["p1"]["angles"] = np.array(starfile.loc[:, ["pairAnglePhi1", "pairAnglePsi1", "pairAngleTheta1"]])
    data_dict["p1"]["classes"] = starfile["pairClass1"].values
    data_dict["p1"]["tomoName"] = starfile["pairTomoName"].values
    data_dict["p1"]["psfs"] = starfile["pairPsf1"].values
    data_dict["p1"]["pixs"] = starfile["pairPixelSizeAng"].values
    data_dict["p1"]["orgListIDX"] = starfile["pairIDX1"].values
    data_dict["p1"]["pairPosInPoly"] = starfile["pairPosInPoly1"].values

    data_dict["p2"]["positions"] = np.array(starfile.loc[:,["pairCoordinateX2","pairCoordinateY2","pairCoordinateZ2"]])
    data_dict["p2"]["angles"] = np.array(starfile.loc[:, ["pairAnglePhi2", "pairAnglePsi2", "pairAngleTheta2"]])
    data_dict["p2"]["classes"] = starfile["pairClass2"].values
    data_dict["p2"]["tomoName"] = starfile["pairTomoName"].values
    data_dict["p2"]["psfs"] = starfile["pairPsf2"].values
    data_dict["p2"]["pixs"] = starfile["pairPixelSizeAng"].values
    data_dict["p2"]["orgListIDX"] = starfile["pairIDX2"].values
    data_dict["p2"]["pairPosInPoly"] = starfile["pairPosInPoly2"].values
    
    data_dict["label"]["pairClass"] = starfile["pairClass"].values
    data_dict["label"]["pairClassColour"] = np.zeros([starfile.shape[0],3])
    for i in range(starfile.shape[0]):
        single_pairClassColour = [float(j) for j in starfile["pairClassColour"].values[i].split("-")]
        #print(single_pairClassColour)
        data_dict["label"]["pairClassColour"][i,:] = single_pairClassColour
        cmbInd[i,3:6] = single_pairClassColour
    data_dict["label"]["pairLabel"] = starfile["pairLabel"].values
    data_dict["label"]["tomoName"] = starfile["pairTomoName"].values
    data_dict["label"]["tomoID"] = np.array([-1]*starfile.shape[0],dtype = np.int )
    data_dict["label"]["p1p2TransVect"] = data_dict["p2"]["positions"] - data_dict["p1"]["positions"]
    data_dict["label"]["orgListName"] = starfile["pairOriPartList"].values
    
    cmbInd[:,0] = starfile["pairIDX1"].values
    cmbInd[:,1] = starfile["pairIDX2"].values
    cmbInd[:,2] = starfile["pairClass"].values
    cmbInd[:,6] = starfile["pairLabel"].values
    
    data_dict["path"] = cmbInd
    
    return data_dict   

def extractFromRelionStar2(starfile, pixelSize):
    data_size = starfile.shape[0]
    data_dict = { }
    data_dict["p1"] = { }
    data_dict["label"] = {}
 
    data_dict["p1"]["positions"] = np.array(starfile.loc[:,["rlnCoordinateX","rlnCoordinateY","rlnCoordinateZ"]])       
    data_dict["p1"]["angles"] = np.zeros([starfile.shape[0],3], dtype = np.float)
    for i in range(starfile.shape[0]):
        _, eluer_angles = tom_eulerconvert_xmipp(starfile["rlnAngleRot"].values[i], starfile["rlnAngleTilt"].values[i],
                                                starfile["rlnAnglePsi"].values[i])
        data_dict["p1"]["angles"][i,:] = eluer_angles
    data_dict["p1"]["classes"] = starfile["rlnClassNumber"].values
    data_dict["p1"]["tomoName"] = starfile["rlnMicrographName"].values
    data_dict["p1"]["psfs"] = starfile["rlnCtfImage"].values
    data_dict["p1"]["pixs"] = np.repeat(pixelSize,data_size)
    data_dict["label"]["tomoName"] = starfile["rlnMicrographName"].values
    
    return data_dict 


def extractFromRelionStar3(starfile, pixelSize):
    data_size = starfile.shape[0]
    data_dict = { }
    data_dict["p1"] = { }
    data_dict["label"] = {}
 
    data_dict["p1"]["positions"] = np.array(starfile.loc[:,["rlnCoordinateX","rlnCoordinateY","rlnCoordinateZ"]])       
    data_dict["p1"]["angles"] = np.zeros([starfile.shape[0],3], dtype = np.float)
    for i in range(starfile.shape[0]):
        _, eluer_angles = tom_eulerconvert_xmipp(starfile["rlnAngleRot"].values[i], starfile["rlnAngleTilt"].values[i],
                                                starfile["rlnAnglePsi"].values[i])
        data_dict["p1"]["angles"][i,:] = eluer_angles
    data_dict["p1"]["classes"] = starfile["rlnClassNumber"].values
    data_dict["p1"]["tomoName"] = starfile["rlnMicrographName"].values
    data_dict["p1"]["psfs"] = starfile["rlnCtfImage"].values
    data_dict["p1"]["pixs"] = np.repeat(pixelSize,data_size)
    data_dict["label"]["tomoName"] = starfile["rlnMicrographName"].values
    
    return data_dict

def extractFromStopGapStar(starfile, pixelSize):
    data_size = starfile.shape[0]
    data_dict = { }
    data_dict["p1"] = { }
    data_dict["label"] = {}

    data_dict["p1"]["positions"] = np.array(starfile.loc[:,["orig_x","orig_y","orig_z"]])
    data_dict["p1"]["angles"] = np.array(starfile.loc[:,["phi","psi","the"]])
    data_dict["p1"]["classes"] = starfile["class"].values
    data_dict["p1"]["tomoName"] = np.array(["tomo%d"%int(i) for i in starfile["tomo_num"].values])
    data_dict["p1"]["psfs"] = np.repeat("-1",data_size)
    data_dict["p1"]["pixs"] = np.repeat(pixelSize,data_size)
    data_dict["label"]["tomoName"] = np.array(["tomo%d"%int(i) for i in starfile["tomo_num"].values])
    
    return data_dict
    

   
def updateTomoID(st):
    tomoNames = st["p1"]["tomoName"]
    tomoIDs = np.zeros(len(tomoNames),dtype = np.int)
    utomoName = np.unique(tomoNames)
    len_utomoName = len(utomoName)
    for i in range(len_utomoName):
        idx = np.where(tomoNames == utomoName[i])[0]
        tomoIDs[idx] = i
    st["label"]["tomoID"] = tomoIDs
    return st
