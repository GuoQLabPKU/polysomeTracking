import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

def tom_kdeEstimate(dist1, dist1Label, fTitle = '', save_dir = '', ifDisplay = 1,
                    cdfValue = 0.05, dist2 = '', dist2Label = '', mode = 'basic'):
    '''
    TOM_KDEESTIMATE is aimed to fit a gauss based KDE estimator to one distribution, 
    if more than one distribution is given, two fitting KDE will be returned. And
    the border with cdf = 1-cdfValue. And the p-value of overlapped region of these 
    two distribution
    '''
    
#    colli_dist = {1.0: 56.15207,7.0: 101.19718,9.0: 74.42177,6.0: 131.26727,
#                  5.0: 82.31526,11.0: 144.52849,4.0: 71.12578,8.0: 99.91933,
#                  3.0: 108.729706,2.0: 71.62952,10.0: 81.22351}
#    classNr = int(dist1Label[8:])
    #dist = colli_dist[classNr]
    kde1 = sps.gaussian_kde(dist1)
    if ifDisplay:
        plt.figure()
        plt.hist(dist1, label = dist1Label, density = True, alpha = 0.4)
        #plt.vlines(x = dist, ymin = 0, ymax = 0.09, linewidth = 2, color = 'black')
        plt.ylim([0,0.09])
        spread = np.linspace(np.min(dist1), np.max(dist1), 300)
        plt.plot(spread, kde1.pdf(spread), label = 'KDE_%s'%dist1Label)
    
    #return the border of p-value (0.05 maybe?)
    cdfBorder1, cdf1 = borderCal(kde1, np.min(dist1), 
                               np.max(dist1), 5, cdfValue)
    if len(dist2) > 0:
        kde2 = sps.gaussian_kde(dist2)
        if ifDisplay:
            plt.hist(dist2, label = dist2Label,
                     density = True, alpha = 0.4)   
            spread = np.linspace(np.min(dist2), np.max(dist2))
            plt.plot(spread, kde2.pdf(spread), label = 'KDE_%s'%dist2Label) 
            
            if mode == 'advance':
                #here, we want to show the CDF of these two overlapped regions
                if np.max(dist1) > np.min(dist2):
                    overlapDist1 = kde1.integrate_box_1d(np.min(dist2), np.max(dist1))
                    coordy1 = kde1.pdf(np.min(dist2))
                    plt.text(np.min(dist2), coordy1, 'pValues:%.3f'%overlapDist1)
                
                    overlapDist2 = kde2.integrate_box_1d(np.min(dist2), np.max(dist1))
                    coordy2 = kde2.pdf(np.max(dist1))
                    plt.text(np.max(dist1), coordy2, 'pValues:%.3f'%overlapDist2)
                
                elif np.max(dist2) > np.min(dist1):
                    overlapDist1 = kde1.integrate_box_1d(np.min(dist1), np.max(dist2))
                    coordy1 = kde1.pdf(np.max(dist2))
                    plt.text(np.max(dist2), coordy1, 'pValues:%.3f'%overlapDist1)
                
                    overlapDist2 = kde2.integrate_box_1d(np.min(dist1), np.max(dist2))
                    coordy2 = kde2.pdf(np.min(dist1))
                    plt.text(np.min(dist1), coordy2, 'pValues:%.3f'%overlapDist2)  
            
        #return the border of distribution2 with p-value
        cdfBorder2, cdf2 = borderCal(kde2, np.min(dist2), 
                               np.max(dist2), 5, cdfValue)
    if ifDisplay:  
        plt.title(fTitle)
        plt.legend(fontsize = 15, edgecolor='black')
        #modify the size of the xticks 
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        #plt.show() 
    if len(save_dir) > 0:
        plt.savefig('%s/%s_%s_%s.png'%(save_dir, dist1Label, dist2Label, fTitle), dpi = 300)
    plt.close()
        
    return cdfBorder1, cdf1, cdfBorder2, cdf2
    
    
def borderCal(kde, lowborder, initborder, cycle_n = 5, pvalue = 0.05 ):
    upperborder = initborder 
    while cycle_n > 0:
        spread = np.linspace(lowborder, upperborder, 300)
        cdfList = np.array([kde.integrate_box_1d(lowborder, i) for i in spread])
        cdfListNearPvalue = cdfList-pvalue
        cdfListNearPvalueAbs = np.abs(cdfListNearPvalue)
        idxmin = np.argmin(cdfListNearPvalueAbs)
        if cdfListNearPvalue[idxmin] > 0:
            upperborder = spread[idxmin]
        else:
            if idxmin < len(spread) - 1:
                upperborder = spread[idxmin+1]
            else:
                return spread[idxmin], cdfListNearPvalue[idxmin]
        cycle_n -= 1
    cdf = kde.integrate_box_1d(lowborder, upperborder) 
    return upperborder, cdf
