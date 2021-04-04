import os
from tom_functions.tom_starread import tom_starread
from tom_functions.tom_calcTransforms import tom_calcTransforms
def calcTransForms(pSt):
    '''
    the input should be a instance of the pSt class.
    with attributes of io directories/transform parameters/classify parameters/sel/vis parameters/clst, 
    each attribute shoule be one dict
    '''
    maxDistInPix = pSt.transForm["maxDist"]/pSt.transForm["pixS"]
    transFormFile = pSt.io["classifyFold"] + "/allTransforms.star"
    if os.access(transFormFile, os.F_OK):
        if os.access(transFormFile, os.R_OK):
            transList = tom_starread(transFormFile)
        else:
            print("Error: the transformation file exists, but can't be read! Check and try again!")
    else:
        transList = tom_calcTransforms(pSt.io["posAngList"], '', maxDistInPix, 'exact', transFormFile)
        
    return transList
