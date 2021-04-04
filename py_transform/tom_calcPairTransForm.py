import numpy as np
from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate
from py_transform.tom_angular_distance import tom_angular_distance

def tom_calcPairTransForm(pos1, ang1, pos2, ang2, dMetric = 'exact'):
    '''
    TOM_CALCPAIRTRANSFORM calculates relative transformation between two poses
                 
    [posTr,angTr]=tom_calcPairTransForm(pos1,ang1,pos2,ang2)
    
    PARAMETERS
    
    INPUT
        pos1                 position 1
        ang1                 angle1 in Z-X-Z
        pos2                 posion2
        ang2                 angle2 in Z-X-Z
        dMetric              ('exact') at the moment only exact implemented
    
    OUTPUT
        posTr               transformation vector between two points (the coordinates(x-y-z) of one pose coordinate system) 
                            one-dimensition array
                            
        angTr               transformation angle between two points
                            one-dimensition array
        
        lenPosTr            length of transformation vector     
        lenAngTr            angular distance from [0 0 0] to angTr (using quaternions)
    
    
    EXAMPLE
       [pos,rot]=tom_calcPairTransForm(np.array([1, 1, 1]),np.array([0, 0, 10]),np.array([2, 2, 2]),np.array([0, 0, 30]));
      
    
    REFERENCES    
    '''
    if dMetric == 'exact':
        ang1Inv = np.array([-ang1[1],-ang1[0],-ang1[2]])
        compare_array = np.zeros([2,3])
        compare_array[0,:] = ang2
        compare_array[1,:] = ang1Inv
        #calculate euler angles of relative rotation
        angTr, _, _ = tom_sum_rotation(compare_array, np.zeros([2,3]))
        #calclulate relative coordinates of pos2 in the ang1-po1 coordinate system
        pos2Rel = pos2-pos1
        posTr = tom_pointrotate(pos2Rel, ang1Inv[0], ang1Inv[1], ang1Inv[2])
        #calculte eluer distance of posTr
        lenPosTr = np.linalg.norm(posTr)
        #calculate the quaternions, from zero angle to relative rotation angle
        lenAngTr = tom_angular_distance(np.zeros(3),angTr)
    
    return posTr, angTr, lenPosTr, lenAngTr   