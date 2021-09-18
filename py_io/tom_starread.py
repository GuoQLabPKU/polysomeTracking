import pandas as pd

def tom_starread(starfile ,pixS = -1):
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
    f = open(starfile,"r+")
    all_line = f.readlines()
    #detect the type of starfile:relion2/relion3/stopgap
    startype = startype_check(all_line)
    if startype == 'relion2':
        starInfos = readRelion2(all_line)
    elif startype == 'relion3':
        starInfos = readRelion3(all_line)        
    elif startype == 'stopgap':
        starInfos = readSG(all_line)      
    
    if pixS > 0:
        starInfos['pixelSize'] = pixS                        
    f.close()
    return starInfos



def startype_check(all_line):
    count = 10
    for single_line in all_line:
        if count > 0:
            if single_line.startswith("data_"):
                if single_line.startswith("data_optics"):
                    return 'relion3'
                if single_line.startswith("data_stopgap"):
                    return 'stopgap'
                else:
                    return 'relion2'
        else:
            raise TypeError('''Unknown type of input star file! 
                            Only relion2/relion3/stopgap star files are supported!''')
        count -= 1
        
def generateStarInfos():
    starInfos = { }
    starInfos['type'] = 'relion2'
    starInfos['pixelSize'] = -1.0
    starInfos['header_particles'] = ['data_', 'loop_']
    starInfos['data_particles'] = None
    
    starInfos['header_optics'] = None
    starInfos['data_optics'] = None
    
    return starInfos
    
def readRelion2(all_line):
    starInfos = generateStarInfos()
    starInfos['header_particles'] = ['data_', 'loop_']
    
    colname_dict = { }
    colname_list = [ ]
    for single_line in all_line:
        if single_line.startswith("_"):
            if "#" in single_line:
                colname, colnum = single_line.split("#")
            else:
                colname = single_line
            colname = colname.rstrip()[1:]
            colname_dict[colname] = [ ]
            colname_list.append(colname)
        elif single_line.startswith("data_"):
            continue
        elif single_line.startswith("loop_"):
            continue
        elif single_line.strip() == '':
            continue
        else:
            line_content = single_line.split(" ")
            line_content_clean = [i  for i in line_content if len(i)!= 0]
            if line_content_clean[-1] == '\n':
                if len(line_content_clean)-1 != colname_list.__len__():  #because the last char shoule be "\n" 
                    raise TypeError("the star data is not consistent!") 
                    #os._exit(1)
            
            else:
                if "\n" in line_content_clean[-1]:
                    if len(line_content_clean) != colname_list.__len__(): 
                        raise TypeError("the star data is not consistent!")
                    else:
                        last_element = line_content_clean[-1]
                        last_element_clean = last_element[0:-1]
                        line_content_clean[-1] = last_element_clean
                else:
                    raise TypeError("newline symbol detected! Check you file !")                       
            for single_element, single_column in zip(line_content_clean, colname_list):
                try:
                    if "." in single_element:
                        single_element = float(single_element)
                    else:
                        single_element = int(single_element)
                except ValueError:
                    pass
                colname_dict[single_column].append(single_element)
    #make particle store dataFrame
    data_particles = pd.DataFrame(colname_dict)    
    starInfos['data_particles'] = data_particles
    if 'rlnDetectorPixelSize' in data_particles.columns:
        starInfos['pixelSize'] = data_particles['rlnDetectorPixelSize'].values[0]
    elif  'pairPixelSizeAng' in  data_particles.columns:
        starInfos['pixelSize'] = data_particles['pairPixelSizeAng'].values[0]
    else:
        pass
    
    return starInfos


def readRelion3(all_line):
    starInfos = generateStarInfos()
    starInfos['type'] = 'relion3'
    starInfos['header_particles'] = ['# version 30001','data_particles','loop_']
    starInfos['header_optics'] = ['# version 30001','data_optics','loop_']
    #read the files 
    colnameParticles_dict = { }
    colnameParticles_list = [ ]
    colnameOptics_dict = { }
    colnameOptics_list = [ ]
    
    #search the range belonging to optics data 
    Optics_range = 0
    for single_line in all_line:
        if single_line.startswith('# version') & (Optics_range > 5):
            break
        Optics_range += 1
    
    #analysis the optics part
    for single_line in all_line[:Optics_range]:
        if single_line.startswith("_"):
            if "#" in single_line:
                colname, colnum = single_line.split("#")
            else:
                colname = single_line
            colname = colname.rstrip()[1:]
            colnameOptics_dict[colname] = [ ]
            colnameOptics_list.append(colname)
        elif single_line.startswith("data_optics"):
            continue
        elif single_line.startswith("loop_"):
            continue
        elif single_line.strip() == '':
            continue
        elif single_line.startswith("# version"):
            continue
        else:
            line_content = single_line.split(" ")
            line_content_clean = [i for i in line_content if len(i)!= 0]
            if line_content_clean[-1] == '\n':
                if len(line_content_clean)-1 != colnameOptics_list.__len__():  #because the last char shoule be "\n" 
                    raise TypeError("the star data is not consistent!") 
                    #os._exit(1)            
            else:
                if "\n" in line_content_clean[-1]:
                    if len(line_content_clean) != colnameOptics_list.__len__(): 
                        raise TypeError("the star data is not consistent!")
                    else:
                        last_element = line_content_clean[-1]
                        last_element_clean = last_element[0:-1]
                        line_content_clean[-1] = last_element_clean
                else:
                    raise TypeError("newline symbol detected! Check you file !")                      
            for single_element, single_column in zip(line_content_clean, colnameOptics_list):
                try:
                    if "." in single_element:
                        single_element = float(single_element)
                    else:
                        single_element = int(single_element)
                except ValueError:
                    pass
                colnameOptics_dict[single_column].append(single_element)
                
    #analysis the particle part
    for single_line in all_line[Optics_range:]:
        if single_line.startswith("_"):
            if "#" in single_line:
                colname, colnum = single_line.split("#")
            else:
                colname = single_line
            colname = colname.rstrip()[1:]
            colnameParticles_dict[colname] = [ ]
            colnameParticles_list.append(colname)
        elif single_line.startswith("data_particles"):
            continue
        elif single_line.startswith("loop_"):
            continue
        elif single_line.strip() == '':
            continue
        elif single_line.startswith("# version"):
            continue
        else:
            line_content = single_line.split(" ")
            line_content_clean = [i  for i in line_content if len(i)!= 0]
            if line_content_clean[-1] == '\n':
                if len(line_content_clean)-1 != colnameParticles_list.__len__():  #because the last char shoule be "\n" 
                    raise TypeError("the star data is not consistent!")            
            else:
                if "\n" in line_content_clean[-1]:
                    if len(line_content_clean) != colnameParticles_list.__len__(): 
                        raise TypeError("the star data is not consistent!")
                    else:
                        last_element = line_content_clean[-1]
                        last_element_clean = last_element[0:-1]
                        line_content_clean[-1] = last_element_clean
                else:
                    raise TypeError("newline symbol detected! Check you file !")
                    #os._exit(1)                        
            for single_element, single_column in zip(line_content_clean, colnameParticles_list):
                try:
                    if "." in single_element:
                        single_element = float(single_element)
                    else:
                        single_element = int(single_element)
                except ValueError:
                    pass
                colnameParticles_dict[single_column].append(single_element)
       
    #make particle store dataFrame
    data_optics = pd.DataFrame(colnameOptics_dict)
    data_particles = pd.DataFrame(colnameParticles_dict)    
    starInfos['data_optics'] = data_optics
    starInfos['data_particles'] = data_particles
    starInfos['pixelSize'] = data_optics['rlnImagePixelSize'].values[0]
    return starInfos
    
def readSG(all_line):
    starInfos = generateStarInfos()
    starInfos['type'] = 'stopgap'
    starInfos['header_particles'] = ['data_stopgap_motivelist', 'loop_']
    
    colname_dict = { }
    colname_list = [ ]
    for single_line in all_line:
        if single_line.startswith("_"):
            if "#" in single_line:
                colname, colnum = single_line.split("#")
            else:
                colname = single_line
            colname = colname.rstrip()[1:]
            colname_dict[colname] = [ ]
            colname_list.append(colname)
        elif single_line.startswith("data_stopgap"):
            continue
        elif single_line.startswith("loop_"):
            continue
        elif single_line.strip() == '':
            continue
        else:
            line_content = single_line.split(" ")
            line_content_clean = [i  for i in line_content if len(i)!= 0]
            if line_content_clean[-1] == '\n':
                if len(line_content_clean)-1 != colname_list.__len__():  #because the last char shoule be "\n" 
                    raise TypeError("the star data is not consistent!") 
                    #os._exit(1)
            
            else:
                if "\n" in line_content_clean[-1]:
                    if len(line_content_clean) != colname_list.__len__(): 
                        raise TypeError("the star data is not consistent!")
                    else:
                        last_element = line_content_clean[-1]
                        last_element_clean = last_element[0:-1]
                        line_content_clean[-1] = last_element_clean
                else:
                    raise TypeError("newline symbol detected! Check you file !")                       
            for single_element, single_column in zip(line_content_clean, colname_list):
                try:
                    if "." in single_element:
                        single_element = float(single_element)
                    else:
                        single_element = int(single_element)
                except ValueError:
                    pass
                colname_dict[single_column].append(single_element)
    #make particle store dataFrame
    data_particles = pd.DataFrame(colname_dict)    
    starInfos['data_particles'] = data_particles
    return starInfos    
    
  