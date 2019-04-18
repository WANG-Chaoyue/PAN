from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.layers import Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid,tanh, softmax
from theano.tensor import TensorType
from lasagne.init import Normal, HeNormal
from time import time
import scipy.io as scio
import sys
import os
#from pylearn2.gui.patch_viewer import PatchViewer
from PIL import Image
from matplotlib.pyplot import imshow, imsave, imread

import numpy as np
import theano
import theano.tensor as T
import json
import lasagne

from datasets import pix2pix,Inpainting
from lib.data_utils import processing_img, convert_img_back, convert_img, pix2pixBatch, shuffle, iter_data, inpainting_data

import models

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main():
    # Parameters
    train_data = './datasets/facades/test/'
    JPEG_img = False 
    start = 0
    stop = 106 
    save_samples = True 
    h5py = 0
    batchSize = 4
    loadSize = 256 
    fineSize = 256 
    flip = False
    ngf = 64
    ndf = 64
    input_nc = 3
    output_nc = 3
    epoch = 1000 

    task = 'facades'
    name = 'nogan'
    which_direction = 'BtoA'
    preprocess = 'regular'
    which_netG = 'unet_nodrop'
    test_deterministic = True   
    
    patchSize = 64
    overlap = 4
    
    desc = task + '_' + name 
    print desc

    # Load the dataset
    print("Loading data...")
    if h5py == 0 and preprocess == 'regular':
        if which_direction == 'AtoB':
            test_input, test_output, file_name = pix2pix(data_path=test_data, img_shape=[input_nc,loadSize,loadSize], save = save_samples, start=start, stop=stop)
        elif which_direction == 'BtoA':
            test_output, test_input, file_name = pix2pix(data_path=test_data, img_shape=[input_nc,loadSize,loadSize], save = save_samples, start=start, stop=stop)
        ids = range(0,stop-start)  
    elif h5py == 1 and preprocess == 'regular':
        print('waiting to fill')
        ids = range(start,stop)  
    elif h5py == 0 and task == 'inpainting':
        test_input, file_name = Inpainting(data_path=test_data, img_shape=[input_nc,loadSize,loadSize], save = save_samples, start=start, stop=stop)
	test_output = test_input
	ids = range(0,stop-start)  
    elif h5py == 1 and task == 'inpainting':
        print('waiting to fill')
        ids = range(start,stop)  
    elif h5py == 0 and task == 'cartoon':
        print('waiting to fill')
        ids = range(0,stop-start)  
    elif h5py == 1 and task == 'cartoon':
        print('waiting to fill')
        ids = range(start,stop)  
    
    ntrain = len(ids)
   
    # Prepare Theano variables for inputs and targets
    input_x = T.tensor4('input_x')
    input_y = T.tensor4('input_y')
    
    # Create neural network model
    print("Building model and compiling functions...")
    if which_netG == 'unet':
        generator = models.build_generator_unet(input_x,ngf=ngf)
    elif which_netG == 'unet_nodrop':
        generator = models.build_generator_unet_nodrop(input_x,ngf=ngf)
    elif which_netG == 'unet_1.0':
        generator = models.build_generator_unet_1(input_x,ngf=ngf)
    elif which_netG == 'unet_deraining':
        generator = models.build_generator_deraining(input_x,ngf=ngf)
    elif which_netG == 'Ginpainting':
        generator = models.build_generator_inpainting(input_x,ngf=ngf)
    else:    
	print('waiting to fill')

    with np.load('models/%s/gen_%d.npz'%(desc,epoch)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator, param_values)

    gen_fn = theano.function([input_x],
                             lasagne.layers.get_output(generator, deterministic=test_deterministic))
    
    test_folder = desc +'_'+ str(epoch)
    real = True

    if not os.path.isdir('test_imgs/'+test_folder):
        os.mkdir(os.path.join('test_imgs/',test_folder))
    test_folder_path = str('test_imgs/' + test_folder + '/')
    test_folder_path_patch = str(test_folder_path+'output/') 
    test_folder_path_image = str(test_folder_path+'input/') 
    test_folder_path_real = str(test_folder_path+'GroundTruth/') 
    if not os.path.isdir(test_folder_path_patch):
        os.mkdir(os.path.join(test_folder_path_patch))
    if not os.path.isdir(test_folder_path_image):
        os.mkdir(os.path.join(test_folder_path_image))
    if not os.path.isdir(test_folder_path_real):
        os.mkdir(os.path.join(test_folder_path_real))
    i = 1  
    for index in iter_data(ids, size=batchSize):
        xmb = test_input[index,:,:,:] 
        ymb = test_output[index,:,:,:] 
        if preprocess == 'regular':
            xmb, ymb = pix2pixBatch(xmb,ymb,fineSize,input_nc,flip=flip)
        elif task == 'inpainting':
            dmb,_ = pix2pixBatch(xmb,ymb,fineSize,input_nc,flip=flip)
	    xmb, ymb = inpainting_data(dmb, image_shape=[input_nc,fineSize,fineSize], patch_shape=[input_nc,patchSize,patchSize],overlap=overlap)
        elif task == 'cartoon':
            print('waiting to fill')
	    
        xmb = processing_img(xmb,convert=False) 
        ymb = processing_img(ymb,convert=False)
	images = gen_fn(xmb)
        for ii in xrange(images.shape[0]):
	    idd = index[ii]
	    ff = file_name[idd]
	    if JPEG_img is True:
	        fff = ff[:-5] + '.png'
	    elif JPEG_img is False: 
	        fff = ff[:-4] + '.png'
            
	    img = images[ii]
            img_real = ymb[ii,:,:,:]
            img_whole = xmb[ii,:,:,:]
	    if preprocess == 'regular': 
	        img_whole = convert_img_back(img_whole)
                img = convert_img_back(img)
                img_real = convert_img_back(img_real)
            elif preprocess == 'inpainting':
		img0 = np.zeros([input_nc,fineSize,fineSize],dtype='float32')
		img_real0 = np.zeros([input_nc,fineSize,fineSize],dtype='float32')
		img_whole0 = np.zeros([input_nc,fineSize,fineSize],dtype='float32')
	        
		img0[:,:,:] = img_whole[:,:,:]
		img0[:,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap] = img[:,overlap:patchSize-overlap,overlap:patchSize-overlap:]

		img_real0[:,:,:] = img_whole[:,:,:]
		img_real0[:,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap] = img_real[:,overlap:patchSize-overlap,overlap:patchSize-overlap:]

		img_whole0[:,:,:] = img_whole[:,:,:]
		img_whole0[:,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap,(fineSize-patchSize)/2+overlap:(fineSize+patchSize)/2-overlap] = 1.0 
	        img_whole = convert_img_back(img_whole0)
                img = convert_img_back(img0)
                img_real = convert_img_back(img_real0)
	    
	    result_img = Image.fromarray(((img+1) / 2 * 255).astype(np.uint8))
            result_img.save(test_folder_path_patch + fff)
	    result = Image.fromarray(((img_whole+1) / 2 * 255).astype(np.uint8))
            result.save(test_folder_path_image + fff)
            result_real = Image.fromarray(((img_real+1) / 2 * 255).astype(np.uint8))
            result_real.save(test_folder_path_real + fff)
            i += 1

if __name__ == '__main__':
    '''
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
    '''
    main()
