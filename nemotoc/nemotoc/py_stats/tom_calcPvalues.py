import scipy
import scipy.stats

def tom_calcPvalues(statistics, dist_name, param, alternative = 'greater'):
    '''
    TOM_CALCPVALUES is aimed to return the p-value gived one statstics of one
    specific distribution
    
    
    EXAMPLE
    transList = tom_calcPvalues(1.5, 'norm',(0,1));
    
    PARAMETERS
    
    INPUT
        statistics       statistics  (1D array)     
        dist_name        the type of distribution, only ['expon',
                         'gamma', 'lognorm', 'norm']  are offered
        param            the parameters of given distribution like loc,scale
        alternative      ('greater',opt), greater, less, two-sides
        
                         
    OUTPUT
        p_values         the hypothesis testing p-values (1D array)    
        
    '''
    
    dist_names = [
                  'expon',
                  'gamma',
                  'lognorm',
                  'norm']
    if isinstance(dist_name, str):
         if dist_name not in dist_names:
             raise TypeError('Not a recognized distribution model, \
                             only %s are permitted.'%str(dist_names))
    else:
        raise TypeError('Not a recognized distribution model, \
                        only %s are permitted.'%str(dist_names))       
   
    dist = getattr(scipy.stats, dist_name)
    cumdf = dist.cdf(statistics, *param[:-2], loc=param[-2], 
                              scale=param[-1])
    if alternative == 'greater':
        pvalues = 1-cumdf
    elif alternative == 'less':
        pvalues = cumdf
    else:
        pvalues = 2*(1-cumdf)
    return pvalues
    