import optparse 
import os
import sys
import shutil

import nemotoc

def genConf():
    parse=optparse.OptionParser(usage='"Usage:nemotocGen [--getConf|--getTestData]"')  
    parse.add_option('-c','--getConf',dest='getConf', type=int, help='fetch configure file. 1:yes/0:no')  
    parse.add_option('-t','--getTestData',dest='getTest', type=int, help='fetch test star file. 1:yes/0:no')
    options,args=parse.parse_args()
    
    nemoPath = nemotoc.__path__[0]
    if options.getConf:     
        confFile = '%s/conf.py'%nemoPath
        if os.path.exists(confFile):
            shutil.copyfile(confFile, 'conf.py')
            print('the configure file is named conf.py')
        else:
            print('can not find %s. Can download conf.py from the github'%confFile)
            sys.exit(-1)

    else:
        print('no configure file will be fetched.')
        
    if options.getTest:
        testFile = '%s/data/all_particles_neuron_warp.star'%nemoPath
        if os.path.exists(testFile):
            shutil.copyfile(testFile, 'all_particles_neuron_warp.star')
            print('the test star file is named all_particles_neuron_warp.star')
        else:
            print('can not find %s. Can download all_particles_neuron_warp.star from the github'%confFile)
            sys.exit(-1)

    else:
        print('no all_particles_neuron_warp.star file will be fetched.')

if __name__ == '__main__': 
    genConf()
    
    

