# Functions for signal/image processing coded here in raw python/numpy

import numpy as np

class padding1D:
    def __init__(self,data,padsize):
        self.data = data
        self.padsize = padsize
        
    def zeropad(self):
        paddedimg = np.zeros(shape = (self.data.shape[0]+2*self.padsize[0]),
                             dtype = self.data.dtype)
        paddedimg[self.padsize[0]:self.padsize[0]+self.data.shape[0]] = self.data
        return paddedimg

class padding2D:
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

class kernels1D:
    def __init__(self,kernelsize=(3,)):
        if type(kernelsize) == type(tuple):
            self.kernelsize = kernelsize
        else:
            self.kernelsize = tuple(list(kernelsize))  
        
    def meankernel(self):
        ker = (1/np.prod(self.kernelsize)) * np.ones(shape=self.kernelsize)
        return ker
    
    def identitykernel(self):
        ker = np.zeros(shape=self.kernelsize)
        center = np.array( [(self.kernelsize[0]-1)/2] , dtype='int')
        ker[center] = 1
        return ker
    
    def gaussiankernel(self,mu,sigma):
        ker = np.zeros(shape=self.kernelsize)
        center = np.array( [(self.kernelsize[0]-1)/2] , dtype='int')

    def sharpeningkernel(self):
        pass

class kernels2D:
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
class convolution:
    def __init__(self,data):
        self.data = data

    def convolution1D(self, kernel='identity',kernelsize=(3,),
                      kerneldef=None, padtype='zeropad', stride=(1,)):
        
        stride = tuple(np.array(stride,dtype='int'))

        # Specify kernel:
        if kernel == 'mean':
            ker = kernels1D(kernelsize=kernelsize).meankernel()
        elif type(kerneldef) != type(None):
            # if a user-defined kernel, override defaults
            ker = np.array(kerneldef) 
            kernelsize = ker.shape
        else:
            # if no kernel defined, use identity kernel
            ker = kernels1D(kernelsize=kernelsize).identitykernel()

        # Specify padding
        padsize = (np.array(kernelsize) - np.ones(shape=np.shape(kernelsize)))/2
        padsize = tuple(np.array(padsize,dtype='int'))
            
        if padtype == 'zeropad':
            paddeddata = padding1D(data=self.data,padsize=padsize).zeropad()
        else:
            # if no paddingtype specified, use zero padding
            paddeddata = padding1D(data=self.data,padsize=padsize).zeropad()

        # Convolution
        out = np.zeros(shape=self.data.shape, dtype='float')
        submatrix = np.zeros(shape=ker.shape, dtype='float')
        for n in range(0,out.shape[0],stride[0]):
            submatrix = paddeddata[n:n+ker.shape[0]]
            out[n] = np.sum(np.multiply(submatrix,ker))
        
        # eliminate unnecessary rows and columns skipeed during striding
        idx_filled = np.arange(0,out.shape[0],stride[0])
        out = out[idx_filled]        
        return out

    def convolution2D(self, kernel='identity', kernelsize=(3,3), 
                      kerneldef=None, padtype='zeropad', stride=(1,1)):
        
        stride = tuple(np.array(stride,dtype='int'))
        
        # Specify kernel:
        if kernel == 'mean':
            ker = kernels2D(kernelsize=kernelsize).meankernel()
        elif type(kerneldef) != type(None):
            # if a user-defined kernel, override defaults
            ker = np.array(kerneldef)
            kernelsize = ker.shape
        else:
            # if no kernel defined, use identity kernel
            ker = kernels2D(kernelsize=kernelsize).identitykernel()
        
        # Specify padding
        padsize = (np.array(kernelsize) - np.ones(shape=np.shape(kernelsize)))/2
        padsize = tuple(np.array(padsize,dtype='int'))
            
        if padtype == 'zeropad':
            paddeddata = padding2D(data=self.data,padsize=padsize).zeropad()
        else:
            # if no paddingtype specified, use zero padding
            paddeddata = padding2D(data=self.data,padsize=padsize).zeropad()
        
        # Convolution
        out = np.zeros(shape=self.data.shape, dtype='float')
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
class filter1D:
    def __init__(self, data, kernelsize=(3,), stride=(1,), padtype='zeropadding'):
        self.data = data
        self.kernelsize = kernelsize
        self.stride = tuple(np.array(stride,dtype='int'))
        self.padtype = padtype
    
    def meanfilter(self):
        out = convolution(self.data).convolution1D(kernel='mean', kernelsize=self.kernelsize,
                                                   padtype=self.padtype, stride=self.stride)
        return out

    def gaussianfilter(self):
        pass

    def medianfilter(self):
        pass

    def sharpeningfilter(self):
        pass

class filter2D:
    def __init__(self, data, kernelsize=(3,3), stride=(1,1), padtype='zeropadding'):
        self.data = data
        self.kernelsize = kernelsize
        self.stride = tuple(np.array(stride,dtype='int'))
        self.padtype = padtype
    
    def meanfilter(self):
        out = convolution(self.data).convolution2D(kernel='mean', kernelsize=self.kernelsize,
                                                   padtype=self.padtype, stride=self.stride)
        return out

    def gaussianfilter(self):
        pass

    def medianfilter(self):
        pass

    def sharpeningfilter(self):
        pass

class fourier:
    def __init__(self,data):
        self.data = data
        
    def dft2D(self):
        pass

    def fft2D(self):
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

class wavelets:
    def __init__(self,data):
        pass

class pdf1D:
    def __init__(self,data):
        self.data = data
    
    def kde(self,kernel='gaussian',bw='silverman',weights=None):
        pass

    def histogram(self,bins=100,normed=False,weights=None):
        pass

class pdf_divergence_1D:
    def __init__(self,p,q,px,qx):
        self.p = p
        self.q = q
        self.px = px
        self.qx = qx

    def KLdivergence(self):
        # Gives KL divergence: D_KL(p||q)
        
        xmin = min(min(self.px),min(self.qx))
        xmax = max(max(self.px),max(self.qx))
        num_interp_points = max(len(list(np.array(self.px))) , len(list(np.array(self.qx))) )
        interp_points = np.linspace(xmin,xmax,num_interp_points)
        
        p_interp = np.interp(interp_points,self.px,self.p)
        q_interp = np.interp(interp_points,self.qx,self.q)
        
        Dkl = np.sum(np.multiply(p_interp, np.log(np.divide(p_interp,q_interp))))
        return Dkl

    def JSdivergence(self):
        xmin = min(min(self.px),min(self.qx))
        xmax = max(max(self.px),max(self.qx))
        num_interp_points = max(len(list(np.array(self.px))) , len(list(np.array(self.qx))) )
        interp_points = np.linspace(xmin,xmax,num_interp_points)
        
        p_interp = np.interp(interp_points,self.px,self.p)
        q_interp = np.interp(interp_points,self.qx,self.q)
        
        m = (1./2.) * (p_interp + q_interp)
        
        Dkl_pm = pdf_divergence_1D(p_interp,m,interp_points,interp_points).KLdivergence()
        Dkl_qm = pdf_divergence_1D(q_interp,m,interp_points,interp_points).KLdivergence()
        
        Djs = (1./2.)*Dkl_pm + (1./2.)*Dkl_qm
        
        return Djs
