import numpy as np

def tom_calc_packages(num_of_nodes, num_of_calculations, index = None):
   '''
   TOM_CALC_PACKAGES 

   packages = tom_calc_packages(num_of_nodes,num_of_calculations,index)

   PARAMETERS

   INPUT
       num_of_nodes        number of cpus
       num_of_calculations number of calculations
       index               ...
     
   OUTPUT
       packages            ...

   EXAMPLE
   ... = tom_calc_packages(...)
   
   '''
   if index == None:
       index = np.ones([num_of_nodes,1])
   
   package_size = np.floor(num_of_calculations/num_of_nodes).astype(np.uint32)
   package_rest = np.mod(num_of_calculations,num_of_nodes)
   
   start = 0
   zz = 1
   packages = np.zeros([num_of_nodes, 3],dtype = np.uint32)
   for i in range(num_of_nodes):
       if (num_of_nodes-i) == package_rest:
           package_size += 1
       if index[i] == 1:
           packages[i,:] = np.array([start, start+package_size, package_size])
           zz += 1
           
       start = start + package_size
        
   return packages
           
