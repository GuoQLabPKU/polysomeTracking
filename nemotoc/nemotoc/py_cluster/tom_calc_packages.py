import numpy as np

def tom_calc_packages(num_of_chunck, num_of_calculations, start = 0,index = None):
   '''
   TOM_CALC_PACKAGES 

   packages = tom_calc_packages(num_of_chunck,num_of_calculations,index)

   PARAMETERS

   INPUT
       num_of_chunck        number of chuncks
       num_of_calculations number of calculations
       start                the sites in data, useful when split part of data
       index               ...
     
   OUTPUT
       packages            ...

   EXAMPLE
   ... = tom_calc_packages(...)
   
   '''
   if index == None:
       index = np.ones([num_of_chunck,1])
   
   package_size = np.floor(num_of_calculations/num_of_chunck).astype(np.uint64)
   package_rest = np.mod(num_of_calculations,num_of_chunck)
   
   start = start

   packages = np.zeros([num_of_chunck, 3],dtype = np.uint64)
   for i in range(num_of_chunck):
       if (num_of_chunck-i) == package_rest:
           package_size += 1
       if index[i] == 1:
           packages[i,:] = np.array([start, start+package_size, package_size])
        
           
       start = start + package_size
        
   return packages
           
