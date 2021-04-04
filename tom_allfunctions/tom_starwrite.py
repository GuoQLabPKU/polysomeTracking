import os
import pandas as pd
def tom_starwrite(outputName, st_input, header):
    '''
    TOM_STARWRITE write a dataframe/dict to a star file
    pruneRad(maxDist) to reduce calc (particles with distance > maxDist will not paired)

    transList=tom_calcTransforms(pos,pruneRad,dmetric,verbose)

    PARAMETERS

    INPUT
        outputName          the name of the output files(the default directory is current pathway)
        st_input            the input file, can be a dict or a dataframe
        header              the colnames of the input dataframe or the keys of the dict

    REFERENCES
    
    '''
    #check the type of input data 
    input_type = type(st_input)
    if input_type.__name__ == 'dict':
        print("The input data is one dict.")
        store_data = pd.DataFrame()
        for single_key in header["fieldNames"]:
            store_data[single_key] = st_input[single_key]
            
    elif input_type.__name__ == 'DataFrame':
        print("The input data is one dataframe.")
        store_data = st_input
    
    else:
        print('Error: unrecognized data type!')
        os._exit(1)
    
    #write out the data 
    f = open(outputName,'w')
    f.write(header["title"] + '\n')
    if header['is_loop'] == 1:
        f.write('loop_'+'\n')
    for single_element in header['fieldNames']:
        f.write(single_element + '\n')
    f.close()
    #write the data body
    store_data.to_csv(outputName,mode = 'a', index=False, header = False, sep = " ")
    print("Store the data successfully!")
    

    
    
    
    
    
        