import numpy as np

# Functions for image processing coded here

class padding:
    def __init__(self,data,padsize):
        self.data = data
        self.padsize = padsize
        
    def zeropad(self):
        paddedimg = np.zeros(shape = (self.data.shape[0]+2*self.padsize[0] , 
                                      self.data.shape[1]+2*self.padsize[1]),
                             dtype = self.data.dtype)
        paddedimg[self.padsize[0]:self.padsize[0]+self.data.shape[0],
                  self.padsize[1]:self.padsize[1]+self.data.shape[1]] = self.data
        return paddedimg

class kernels:
    def __init__(self,kernelsize=(3,3)):
        if type(kernelsize) == type(tuple):
            self.kernelsize = kernelsize
        else:
            self.kernelsize = tuple(list(kernelsize))  
        
    def meankernel(self):
        ker = (1/np.prod(self.kernelsize)) * np.ones(shape=self.kernelsize)
        return ker
    
    def identitykernel(self):
        ker = np.zeros(shape=self.kernelsize)
        center = np.array( [(self.kernelsize[0]-1)/2 ,
                            (self.kernelsize[1]-1)/2] , dtype='int')
        ker[center[0],center[1]] = 1
        return ker
    
    def gaussiankernel(self):
        pass

    def sharpeningkernel(self):
        pass

# Define convolution in 2D using plain numpy
def convolution2D(data, kernel='identity', kernelsize=(3,3), 
                  kerneldef=None,
                  padtype='zeropad', stride=(1,1)):
    stride = tuple(np.array(stride,dtype='int'))
    
    # Specify kernel:
    if kernel == 'mean':
        ker = kernels(kernelsize=kernelsize).meankernel()
    elif type(kerneldef) != type(None):
        # if a user-defined kernel, override defaults
        ker = np.array(kerneldef)
        kernelsize = ker.shape
    else:
        # if no kernel defined, use identity kernel
        ker = kernels(kernelsize=kernelsize).identitykernel()
    
    # Specify padding
    padsize = (np.array(kernelsize) - np.ones(shape=np.shape(kernelsize)))/2
    padsize = tuple(np.array(padsize,dtype='int'))
        
    if padtype == 'zeropad':
        paddeddata = padding(data=data,padsize=padsize).zeropad()
    else:
        # if no paddingtype specified, use zero padding
        paddeddata = padding(data=data,padsize=padsize).zeropad()
    
    # Convolution
    out = np.zeros(shape=data.shape, dtype='float')
    submatrix = np.zeros(shape=ker.shape, dtype='float')
    for n in range(0,out.shape[0],stride[0]):
        for m in range(0,out.shape[1],stride[1]):
            submatrix = paddeddata[n:n+ker.shape[0],m:m+ker.shape[1]]
            out[n,m] = np.sum(np.multiply(submatrix,ker))
    
    # eliminate unnecessary rows and columns skipeed during striding
    rows_filled = np.arange(0,out.shape[0],stride[0])
    cols_filled = np.arange(0,out.shape[1],stride[1])
    out = out[rows_filled,:]
    out = out[:,cols_filled]
    
    return out
    
# Define filters
class filter2D:
    def __init__(self, data, kernelsize=(3,3), stride=(1,1)):
        self.data = data
        self.kernelsize = kernelsize
        self.stride = stride = tuple(np.array(stride,dtype='int'))
    
    def meanfilter(self):
        out = convolution2D(data=self.data, kernel='mean', 
                            kernelsize=(3,3),
                            padtype='zeropadding', stride=self.stride)
        return out

    def gaussianfilter(self):
        pass

    def medianfilter(self):
        pass

    def sharpeningfilter(self):
        pass
    
    
def dft2D(data):
    pass

def fft2D(data):
    pass

class spectralestimator2D:
    def __init__(self, data, window):
        self.data = data
        self.window = window 

    def psd_periodogram(self):
        pass

    def psd2D_bartlett(self):
        pass

    def psd2D_welch(self,overlap):
        
        pass

