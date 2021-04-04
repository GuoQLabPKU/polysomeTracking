from py_io import tom_starread
def tom_linkTransforms(pairList, outputName, branchDepth):
    '''
    TOM_FIND_CONNECTEDTANSFORMS finds connedted transform 

    pairs=tom_find_connectedTransforms(pairList,outputName)

    PARAMETERS

    INPUT
       pairList               pari star file                  
       outputName       (opt.) name of the output pair star file
       branchDepth     (1) (opt.) Depth for branch tracing
       
    OUTPUT
       pairListAlg          aligned pair list

    EXAMPLE

    REFERENCES
    
    '''
    if type(pairList).__name__ == 'str':
        pairList = tom_starread(pairList)
    allClass
        
        