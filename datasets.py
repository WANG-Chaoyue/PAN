import os
import sys
import numpy as np
import h5py
from matplotlib.pyplot import imshow, imsave
from PIL import Image

from lib.data_utils import convert_img, convert_img_back

def pix2pix(data_path='/home/chaoywan/Downloads/pix2pix_data/cityscapes/train/',
            img_shape=[3,256,256], save = False,
            start=0, stop=200):
    C, H, W = img_shape
    files_name = os.listdir(data_path)

    X = np.zeros(((stop-start), C, H, W), dtype='uint8')
    y = np.zeros(((stop-start), C, H, W), dtype='uint8')
    for i in xrange(start,stop):
        z = 0
        img = Image.open(data_path + files_name[i])
	
	#if img.size[1] != H: 
	#    img = img.resize((2*W, H),Image.ANTIALIAS)
	img = img.resize((2*W, H),Image.ANTIALIAS)
        img = np.array(img)
	
	if i == 0 and save is True:
            imsave('other/%s'%files_name[i],img,format='jpg') 
	
	if len(np.shape(img)) < C:
            img = [img]*C
	    img = np.array(img)
	    img = img.reshape([C,H,W*2])
	    img = convert_img_back(img) 
		
	img_input = img[:,0:W,:]
        img_real = img[:,W:,:]
       
	if i == 0 and save is True:
            imsave('other/A_%s'%(files_name[i]),img_input,format='jpg')
            imsave('other/B_%s'%(files_name[i]),img_real,format='jpg') 
        
	img_input = convert_img(img_input)
        img_real = convert_img(img_real)

	X[(i-start),:,:,:] = img_input
        y[(i-start),:,:,:] = img_real
        
    return X, y, files_name[start:stop]

def Inpainting(data_path='/home/chaoywan/Downloads/ImageNet/ILSVRC2012_img_test/',
               img_shape=[3,128,128],
               save=True, start=0, stop=100000):
    C, H, W = img_shape
    #Cp, Hp, Wp = patch_shape
    img_size = np.prod(img_shape)
    #patch_size = np.prod(patch_shape)

    #pH1 = int(round((H-Hp)/2))
    #pH2 = int(round((H-Hp)/2))+Hp
    #pW1 = int(round((W-Wp)/2))
    #pw2 = int(round((W-Wp)/2))+Wp

    files_name = os.listdir(data_path)
    files_name.sort(key= lambda x:int(x[-12:-5])) 
    
    X = np.zeros(((stop-start),C, H, W), dtype='uint8')
    #y = np.zeros(((stop-start),C, H, W), dtype='uint8')
    #z = np.zeros(((stop-start),Cp, Hp, Wp), dtype='uint8')

    for i in xrange(start,stop):
        z = 0
        img = Image.open(data_path + files_name[i])
        img = img.resize((W, H),Image.ANTIALIAS)
        img = np.array(img)
        
	if len(np.shape(img)) < 3:
            img = [img]*3
	    img = np.array(img)
	    img = img.reshape([C,H,W])
	    img = convert_img_back(img) 
        
	#patch = img[:, pH1:pH2, pW1:pW2]
        #img_x = img
        #img_x[0, pH1+overlap:pH2-overlap, pW1+overlap:pW2-overlap] = 2*117.0/255.0-1.0
        #img_x[1, pH1+overlap:pH2-overlap, pW1+overlap:pW2-overlap] = 2*104.0/255.0-1.0
        #img_x[2, pH1+overlap:pH2-overlap, pW1+overlap:pW2-overlap] = 2*123.0/255.0-1.0

        if i == 1 and save is True:
            imsave('other/GroundTruth',img,format='png')
            #imsave('patch',patch,format='png')
            #imsave('input',img_x,format='png')
      	
	img = convert_img(img)
        #img_x = convert_img(img_x)
        #patch = convert_img(patch)
        
        X[(i-start),:,:,:] = img 
        #y[(i-start),:,:,:] = img
        #z[(i-start),:,:,:] = patch 
    return X, files_name[start:stop]

def load_images(data_path='/home/chaoywan/Downloads/pix2pix_data/cityscapes/train/',
            img_shape=[3,256,256], save = False,
            start=0, stop=200):
    C, H, W = img_shape
    files_name = os.listdir(data_path)

    X = np.zeros(((stop-start), C, H, W), dtype='uint8')
    for i in xrange(start,stop):
        z = 0
        img = Image.open(data_path + files_name[i])
	
	if img.size[1] != H: 
	    img = img.resize((W, H),Image.ANTIALIAS)
        img = np.array(img)
	
	if i == 0 and save is True:
            imsave('other/%s'%files_name[i],img,format='jpg') 
	
	if len(np.shape(img)) < C:
            img = [img]*C
	    img = np.array(img)
	    img = img.reshape([C,H,W])
	    img = convert_img_back(img) 
		
	img_input = convert_img(img)

	X[(i-start),:,:,:] = img_input
        
    return X, files_name[start:stop]


if __name__ == '__main__':
    start = 0
    stop  = 49825 
    loadSize = 265 
    train_data = '/home/chaoywan/Downloads/pix2pix_data/edges2shoes/train/'
    tra_X, tra_y,_ = pix2pix(data_path=train_data, img_shape=[3,loadSize,loadSize],
    	             save = True, start=start, stop=stop)
    
    f = h5py.File('/home/chaoywan/Downloads/pix2pix_data/hdf5/edges2shoes_train_265.hdf5','w')
    dset = f.create_dataset("A",data=tra_X)
    dset = f.create_dataset("B",data=tra_y)

    #start = 0
    #stop  = 100000 
    #loadSize = 150 
    #train_data = '/home/chaoywan/Downloads/ImageNet/ILSVRC2012_img_test/'
    #tra_X, name = Inpainting(data_path=train_data, img_shape=[3,loadSize,loadSize],
#		    save = True, start=start, stop=stop)
    
    #f = h5py.File('/home/chaoywan/Downloads/pix2pix_data/hdf5/ILSVRC2012_test_150.hdf5','w')
    #dset = f.create_dataset("data",data=tra_X)
