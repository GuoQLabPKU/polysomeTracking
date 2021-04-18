from py_io.tomo_io.image_io import ImageIO as ImageIO
import os 

def load_tomo(fname, mmap = False):
    '''
    
    Load tomogram in disk in numpy format (valid formats: .rec, .mrc, .em)
    :param fname: full path to the tomogram
    :param mmap: if True (default False) a numpy.memmap object is loaded instead of numpy.ndarray, which means that data
        are not loaded completely to memory, this is useful only for very large tomograms. Only valid with formats
        MRC and EM.
        VERY IMPORTANT: This subclass of ndarray has some unpleasant interaction with some operations,
        because it does not quite fit properly as a subclass of numpy.ndarray
    :return: numpy array
    
    '''

    # Input parsing
    stem, ext = os.path.splitext(fname)
    if mmap and (not ((ext == '.mrc') or (ext == '.rec') or (ext == '.em') )):
        print('mmap option is only valid for .mrc, .rec, .em  formats, current ' + ext)
        os._exists(-1)
        
        

    if (ext == '.mrc') or (ext == '.rec'):
        image = ImageIO()
        if mmap:
            image.readMRC(fname, memmap=mmap)
        else:
            image.readMRC(fname)
        im_data = image.data
    elif ext == '.em':
        image = ImageIO()
        if mmap:
            image.readEM(fname, memmap=mmap)
        else:
            image.readEM(fname)
        im_data = image.data
        
    else:
        error_msg = '%s is non valid format.' % ext
        print(error_msg)
        os._exit(-1)
        
    # For avoiding 2D arrays
    if len(im_data.shape) == 2:
        im_data = numpy.reshape(im_data, (im_data.shape[0], im_data.shape[1], 1))

    return im_data

