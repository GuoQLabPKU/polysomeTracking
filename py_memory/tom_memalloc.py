import numpy as np
def tom_memalloc(freeMem = None, worker_n = None, gpu_list = None, chunk = 3000):
    '''
    TOM_MEMALLOC returns the maxChunck for tom_pdist parallel computation 

    maxChunck = tom_memalloc(freeMem,worker_n)

    PARAMETERS

    INPUT
           freeMem        available free memory(MiB)
           worker_n       available cpus 
           gpu_list       availabel gpus (will only use %free memory>50%)

     
    OUTPUT
           maxChunck      the size of rotation matrixes which can be loaded 
                          directory into memory 
                          
    '''
    if isinstance(worker_n, int): #using cpu
        if isinstance(gpu_list, list):
            print('Warning: you can not use cpus and gpus at the same time! \
                   Will use cpus instead.')
        
        import psutil
        cpuN = psutil.cpu_count()
        if cpuN < worker_n:
            print('Warning: given %dcpus exceed all available cpus'%worker_n)
            worker_n = cpuN
            
        mem_free = round(psutil.virtual_memory().free/1024/1024,2)*0.8 #not sure if the psutil is reliable
        if isinstance(freeMem, (int, float)):
            if freeMem > mem_free:
                print('Warning: the given memory is larger than available memory %.2f'%mem_free)
            else:
                mem_free = freeMem
        maxChunk = np.uint64(mem_free*chunk/worker_n) #if memory error, reduce this number(6500),uint64
        print('Free memory: %.2f mib'%mem_free)
        return maxChunk
    else: 
        if isinstance(gpu_list, list):
            import pynvml
            pynvml.nvmlInit()
            gpuN = pynvml.nvmlDeviceGetCount()
            freeMem_dict = {}
            if len(gpu_list) > gpuN:
                print("Warning: the given # gpus exceed available gpus")
                gpu_list = range(gpuN)
            for gpu_id in gpu_list:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = meminfo.total
                free = meminfo.free
                if free/total < 0.5:
                    print('gpu %d already used, skipping'%gpu_id)
                    continue
                freeMem_dict[gpu_id] =  round(free/1024/1024,2)*0.8 #not sure if the pynvml is reliable
                print('GPU %d:%.2f mib free memory'%(gpu_id, round(free/1024/1024,2)*0.8))
            if len(freeMem_dict) == 0:
                raise RuntimeError('Invalid gpu list')
                
            maxChunk_dict = {}
            for gpu_id in freeMem_dict.keys():
                maxChunk_dict[gpu_id] = np.uint64(freeMem_dict[gpu_id]*chunk) #if memory error, reduce this number(6500),uint64
            return maxChunk_dict
                
                
                