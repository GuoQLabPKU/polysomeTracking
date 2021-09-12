import matplotlib.pyplot as plt
import numpy as np

def tom_visDist(distVect, distsAng, distsCN, saveDir, classLabel):  
    plt.figure()
    if len(distVect) > 0:
        plt.hist(distVect,alpha = 0.5, label = 'vect distance')
    if len(distsAng) > 0:
        plt.hist(distsAng,alpha = 0.5, label = 'angle distance')
    if len(distsCN) > 0:
        plt.hist(distsCN, alpha = 0.5, label = 'combined distance')
    plt.legend(fontsize = 15)
    plt.xlabel('Distance between each transformation\nand Tavg',fontsize = 15)
    plt.ylabel('# of transformation',fontsize = 15)
    plt.title('%s\nmean:%.2f, std:%.2f of vect distance\nmean:%.2f, std:%.2f of angle distance\nmean:%.2f, std:%.2f of combined distance'
              %(classLabel, np.mean(distVect),np.std(distVect),np.mean(distsAng),np.std(distsAng),np.mean(distsCN), np.std(distsCN)),
                                                                    fontsize = 8)

    plt.tight_layout()
    if len(saveDir) > 0:
        plt.savefig('%s/%s.png'%(saveDir,classLabel), dpi = 300)
    plt.close()
 

