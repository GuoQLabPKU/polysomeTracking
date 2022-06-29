def tom_starwrite(outputName, star_st):
    '''
    TOM_STARWRITE write a dict to a star file
    PARAMETERS

    INPUT
        outputName          the name of the output files(the default directory is current pathway)
        star_st             the input file, must be a dict following content like tom_starread ouput dict
        
    REFERENCES
    '''
    #check the type of input data 
    support_list = ['relion2', 'relion3', 'stopgap'] 
    assert star_st['type'] in support_list
    writeStarFile(star_st,outputName)
 
def writeStarFile(star_st,outputName): 
    if star_st['type'] == 'relion3':
        writeHeader(star_st['header_optics'], outputName, 1)
        writeDF(star_st['data_optics'], outputName)         
        writeHeader(star_st['header_particles'], outputName,0)
    else:
        writeHeader(star_st['header_particles'], outputName,1)
    writeDF(star_st['data_particles'], outputName)    
         
       
def writeDF(dataFrame, outputName):
    f = open(outputName, 'a+')
    if 'motl' in dataFrame.columns[0]:
        header = ['_%s'%i for i in dataFrame.columns]  
    
    elif ('#' not in dataFrame.columns[0]) & ('motl' not in dataFrame.columns[0]):
        header = [ ]
        for count, single_column in enumerate(dataFrame.columns):
            header.append('_%s #%d'%(single_column, count+1))
    else:
        header = list(dataFrame.columns)
    for single_element in header:
        f.write(single_element + '\n')
    f.close()
    dataFrame.to_csv(outputName, mode = 'a', index = False, header = False, sep = ' ')
            

def writeHeader(header, outputName, overwrite = 1):
    if overwrite:
        f = open(outputName, 'w+')
    else:
        f = open(outputName, 'a+')
    f.write('\n')
    for single_header in header:
        f.write(single_header + '\n')
        if single_header != 'loop_':
            f.write('\n')
    f.close()
