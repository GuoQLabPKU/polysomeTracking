import pandas as pd
import numpy as np
import os
def tom_starread(starfile):
    '''
    tom_starread read star files 
 
    star_st=tom_starread(filename) 
  
    PARAMETERS 
  
    INPUT 
        filename         filename of the star file 
 
 
    OUTPUT 
        star_data        python dataframe  
                          
  
    EXAMPLE 
        star_st=tom_starread('in.star'); 

    REFERENCES 
  
    NOTE: 
  
    '''  
    colname_dict = { }
    colname_list = [ ]
    f = open(starfile,"r+")
    all_line = f.readlines()
    for single_line in all_line:
        if single_line.startswith("_"):
            if "#" in single_line:
                colname, colnum = single_line.split("#")
            else:
                colname = single_line
            colname = colname[1:-1]
            colname_dict[colname] = [ ]
            colname_list.append(colname)
        elif single_line.startswith("data"):
            continue
        elif single_line.startswith("loop"):
            continue
        elif single_line == "\n":
            continue
        else:
            line_content = single_line.split(" ")
            line_content_clean = [i  for i in line_content if len(i)!= 0]
            if line_content_clean[-1] == '\n':
                if len(line_content_clean)-1 != len(colname_dict.keys()):  #because the last char shoule be "\n" 
                    print("Error: the star data is not consistent!") 
                    #os._exit(1)
            
            else:
                if "\n" in line_content_clean[-1]:
                    if len(line_content_clean) != len(colname_dict.keys()):                  
                        print("Error: the star data is not consistent!")
                        break
                        #os._exit(1)
                    else:
                        last_element = line_content_clean[-1]
                        last_element_clean = last_element[0:-1]
                        line_content_clean[-1] = last_element_clean
                else:
                    print("Error: no newline symbol detected! Check you file !")
                    #os._exit(1)                        
            for single_element,single_columne in zip(line_content_clean,colname_list):
                try:
                    if "." in single_element:
                        single_element = float(single_element)
                    else:
                        single_element = int(single_element)
                except ValueError:
                    pass
                colname_dict[single_columne].append(single_element)
    star_data = pd.DataFrame(colname_dict)                                 
    f.close()
    return star_data 