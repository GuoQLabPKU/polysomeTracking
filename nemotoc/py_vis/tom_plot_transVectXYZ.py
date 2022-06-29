import numpy as np 
import matplotlib.pyplot as plt

from nemotoc.py_io.tom_starread import tom_starread

def tom_plot_transVectXYZ(allTransList, statPerClassList, kind, cellLine, classList = None):
    '''
    TOM_PLOT_TRANSVECTXYZ plot the transvect of transformations
    as well as the of the average transformations
    '''
    if isinstance(allTransList, str):
        allTransList = tom_starread(allTransList)
        allTransList = allTransList['data_particles']
    if isinstance(statPerClassList, str):
        statPerClassList = tom_starread(statPerClassList)
        statPerClassList = statPerClassList['data_particles']
        
    if classList is None:
        classList = np.unique(allTransList['pairClass'].values)
    #make figures 
    ax  = plt.figure().gca(projection='3d')
    for singleC in classList:
        transSingle = allTransList[allTransList['pairClass'] == singleC]
        color = transSingle['pairClassColour'].values[0]
        color = [float(i) for i in color.split('-')]
        
        if kind == 'vect':        
            vectX = transSingle['pairTransVectX'].values
            vectY = transSingle['pairTransVectY'].values
            vectZ = transSingle['pairTransVectZ'].values
        else:
            vectX = transSingle['pairTransAngleZXZPhi'].values
            vectY = transSingle['pairTransAngleZXZPsi'].values  
            vectZ = transSingle['pairTransAngleZXZTheta'].values
            if cellLine == 'neuron':
                vectY = [i+360 if i < -100 else i for i in vectY ]      
                if singleC == 7:
                    vectY = [i-360 if i > 150 else i for i in vectY ] 
                    vectX = [i+360 if i < -150 else i for i in vectX]     ###for original neuron                
            else:
                if singleC == 1:
                    vectX = [i+360 if i < 0 else i for i in vectX]
                    vectY = [i+360 if i < 0 else i for i in vectY]
                if singleC == 4:
                    vectX = [i-360 if i > 0 else i for i in vectX]
                    vectY = [i-360 if i > 0 else i for i in vectY]            
                
                if singleC == 3:
                    vectX = [i+360 if i < -100 else i for i in vectX]
                    vectY = [i-360 if i > 100 else i for i in vectY] 
            
#            if singleC == 13:
#                vectX = [i+360 if i < -50 else i for i in vectX]
#                vectY = [i-360 if i > 50 else i for i in vectY] 
#                vectX = [i-360 if i > 250 else i for i in vectX]
#                vectY = [i+360 if i < -250 else i for i in vectY]
#            if singleC == 11:
#                vectY = [i+360 if i < -100 else i for i in vectY] 
#            if singleC == 2:
#                vectY = [i-360 if i > 100 else i for i in vectY] 
#            if singleC == 4:
#                vectY = [i-360 if i > 100 else i for i in vectY] 
#                vectX = [i+360 if i <- 100 else i for i in vectX]
                
                
                
        if cellLine == 'neuron':

            if kind == 'vect':
#                ax.scatter(-48.9880313*0.342, -29.855*0.342, -32.347*0.342,
#                           color = 'black', s =50)
#                ax.scatter(-9.93857248*0.342, -51.12559123*0.342, 39.22695927*0.342,
#                           color = 'cyan', s =50)  
#                ax.scatter(48.9880313*0.342, 29.855*0.342, 32.347*0.342,
#                           color = 'black', s =50) ###==>right direction inv
#                ax.scatter(-0.9880313*0.342, -34.02499533*0.342, 55.61186134*0.342,
#                           color = 'cyan', s =50) ###==>right direction forward 
#                
#
                ax.scatter(np.array(vectX)*0.342, np.array(vectY)*0.342,
                     np.array(vectZ)*0.342, color = np.array(color), 
                     alpha = 1.0, label = 'class%d'%singleC, s = 6)
            else:
#                ax.scatter(-116.1981,  168.9484,   88.7345,
#                           color = 'black', s = 50)
#                ax.scatter(11.0516, -63.8019, 88.7345,
#                           color = 'cyan', s = 50)                 
#                ax.scatter(11.0516, -63.8019, 88.7345,
#                           color = 'black', s = 50)###==>right direction inv
#                ax.scatter(-116.1981,  168.9484,   88.7345,
#                           color = 'cyan', s = 50) ###==>right direction forward    
                                        
                ax.scatter(np.array(vectX), np.array(vectY),
                       np.array(vectZ), color = np.array(color), 
                   alpha = 1.0, label = 'class%d'%singleC, s = 6)
        else:
            if kind == 'vect':
                ax.scatter(np.array(vectX)*0.352, np.array(vectY)*0.352, np.array(vectZ)*0.352,
                       color = np.array(color), alpha = 1.0, s = 6)  
            else:
                ax.scatter(np.array(vectX), np.array(vectY), np.array(vectZ),
                       color = np.array(color), alpha = 1.0, s = 6)                 
        if kind == 'vect':
            ax.set_xlabel('\nx(nm)', fontsize = 15)
            ax.set_ylabel('\ny(nm)', fontsize = 15)
            ax.set_zlabel('\nz(nm)', fontsize = 15)
            ax.set_zticks([-20,0,20])
            ax.set_zticklabels([-20, 0, 20], fontsize = 15) 
            ax.set_xticks([-20, 0,20])
            ax.set_xticklabels([-20,0,20], fontsize = 15)
            ax.set_yticks([-20,0,20])
            ax.set_yticklabels([-20,0,20],fontsize = 15)
        else:
            ax.set_xlabel('\nphi(deg)', fontsize = 15)
            ax.set_ylabel('\npsi(deg)', fontsize = 15)
            ax.set_zlabel('\ntheta(deg)', fontsize = 15)
            ax.set_zticks([0,90,180])
            ax.set_zticklabels([0,90,180], fontsize = 15)   
            
            ax.set_xticks([-180,0,180])
            ax.set_xticklabels([-200,0,200], fontsize = 15)
            ax.set_yticks([-200,0,200])
            ax.set_yticklabels([-200,0,200], fontsize = 15)

            
#            if cellLine == 'ecoli':
#                plt.xticks([-100,0,100],[-100,0,100], fontsize = 15)
#                plt.yticks([-100,0,100], [-100,0,100], fontsize = 15)
#            else:
#                plt.xticks([-180,0,180], [-180,0,180], fontsize = 15) #for orginal neuron
#                plt.yticks([-220,0,220], [-220,0,220], fontsize = 15)  
#                #plt.legend(prop={'size': 12})
    
#    plt.yticks(fontsize = 15)
#    plt.xticks(fontsize = 15)
#    plt.legend(bbox_to_anchor=(1.00, 1),fontsize=15,
#               edgecolor='black')
    ax.grid(False)
    plt.show()