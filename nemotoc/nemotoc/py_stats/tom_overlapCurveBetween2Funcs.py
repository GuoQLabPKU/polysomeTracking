from scipy.integrate import quad

def tom_overlapCurveBetween2Funcs(kde1,kde2, border):
    '''   
    kde1, kde2 from python 
    border:list
    '''
    
    lowerBorder, upperBorder = border
    def y_pts(pt):
        nonlocal kde1, kde2
        y_pt = min(kde1(pt), kde2(pt))
        return y_pt

    overlap = quad(y_pts,lowerBorder, upperBorder )
    return overlap


