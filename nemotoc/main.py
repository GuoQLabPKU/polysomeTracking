import numpy as np
import time
import sys

from nemotoc.py_run.py_run import runPoly
confFile = sys.argv[1]
confFile = confFile[:-3]
conf = __import__(confFile)

if __name__ == '__main__': 
    fillUpPoly = { }
    fillUpPoly['addNum'] = conf.fillUpPoly_addN
    fillUpPoly['fitModel'] = conf.fillUpPoly_model
    fillUpPoly['threshold'] = conf.fillUpPoly_threshold

    avg = { }
    avg['filt'] = { }
    avg['filt']['minNumPart'] = conf.avg_minPart
    avg['filt']['maxNumPart'] = conf.avg_maxPart
    avg['pixS'] = conf.avg_pixS
    avg['maxRes'] = conf.avg_maxRes
    avg['cpuNr'] = conf.avg_cpuN
    avg['callByPython'] = conf.avg_callByPython
    avg['callRelionTemp'] = conf.avg_command
  
    begin = time.time()
    runPoly(conf.input_star, conf.run_folder, conf.project_folder,
            conf.pixel_size, conf.min_dist, conf.subtomo_path, conf.ctf_file,
            conf.search_radius, conf.link_depth, conf.cluster_threshold,
            conf.minNumTransform_ratio, fillUpPoly, conf.cpuN, conf.gpu_list, 
            conf.remove_branches, conf.vectorfield_plotting,
            conf.show_longestPoly, conf.do_avg, avg,
            conf.transNr_initialCluster, conf.iterationNr, conf.do_errorEstimate)
    end = time.time()
    print('Successfully finish NEMO-TOC with %.2fs'%(end-begin))
