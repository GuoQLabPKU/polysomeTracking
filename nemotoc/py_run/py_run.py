import os
import platform

from nemotoc.polysome_class.polysome import Polysome

def runPoly(input_star, run_time, project_folder, pixel_size, min_dist, if_stopgap, subtomo_path, ctf_file,
            search_radius, link_depth, cluster_threshold, minNumTransform_ratio, fillUpPoly, cpuN, gpu_list, remove_branches,
            vectorfield_plotting, show_longestPoly, if_avg, average_particles, avg, transNr, transNr_initialCluster, iterN):
    #check the type of input parameters
    assert isinstance(input_star, str)
    assert isinstance(run_time, str)
    assert isinstance(project_folder, str)
    assert isinstance(pixel_size, (int, float))
    assert isinstance(min_dist, (int, float))
    assert isinstance(if_stopgap, (int, float))
    assert isinstance(subtomo_path, str)
    assert isinstance(ctf_file, str)
    assert isinstance(search_radius, (int, float))
    assert isinstance(link_depth, (int, float))
    assert isinstance(cluster_threshold, (int, float))
    assert isinstance(minNumTransform_ratio, (int, float))
    assert isinstance(fillUpPoly, dict)
    assert isinstance(cpuN, int)
    assert isinstance(gpu_list, (list, type(None)))
    assert isinstance(remove_branches, int)
    assert isinstance(vectorfield_plotting, str)
    assert isinstance(show_longestPoly, int)
    assert isinstance(if_avg, (int, float))
    assert isinstance(average_particles, int)
    assert isinstance(avg, dict)
    #check if the project_folder exist
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)

    polysome1 = Polysome(input_star = input_star, run_time = run_time, proj_folder = project_folder)
    #calculate transformations
    polysome1.transForm['pixS'] = pixel_size
    polysome1.transForm['maxDist'] = search_radius
    polysome1.transForm['branchDepth'] = link_depth
    #do clustering and filtering
    polysome1.classify['clustThr'] = cluster_threshold
    polysome1.sel[0]['minNumTransform'] = minNumTransform_ratio

    polysome1.creatOutputFolder()  #create folder to store the result
    polysome1.preProcess(if_stopgap, subtomo_path, ctf_file, min_dist) #preprocess
    polysome1.calcTransForms(worker_n = cpuN) #calculate transformations
    polysome1.groupTransForms(worker_n = cpuN, gpu_list = gpu_list, transNr = transNr, transNr_initialCluster = transNr_initialCluster, iterN = iterN)  #cluster transformations
    transListSel, selFolds = polysome1.selectTransFormClasses(worker_n = cpuN, gpu_list = gpu_list, iterN = iterN) #filter clusters
    polysome1.genOutputList(transListSel, selFolds) #save the filtered clusters
    polysome1.alignTransforms() #align the transformationsto the same direction
    polysome1.analyseTransFromPopulation('','',1, 0)  #summary the clusters but w/o any polysome information
    polysome1.fillPoly = fillUpPoly #fill up the gaps
    polysome1.link_ShortPoly(remove_branches, cpuN,gpu_list) #link transforms into a long linear chain
    polysome1.analyseTransFromPopulation('','',0, 1) #summary the clusters
    polysome1.noiseEstimate(worker_n = cpuN, gpu_list = gpu_list) #estimate the purity of each cluster

    polysome1.vis['vectField']['type'] = vectorfield_plotting
    polysome1.vis['longestPoly']['render'] = show_longestPoly
    polysome1.visResult()
    polysome1.visLongestPoly()

    #average particles subset using relion_reconstruct
    #detect the operation system type, if windows, skip!
    system = platform.system()
    if system == 'Windows':
        if_avg = 0       
        print('%s platform detected. Particle average commands could not be generated.'%system)
        print('Please find particles at %s/%s/%s for averaging.'%(project_folder, run_time, 'cluster'))
    if if_avg:
        polysome1.avg = avg
        polysome1.generateTrClassAverages()
