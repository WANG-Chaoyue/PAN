from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.layers import Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid,tanh, softmax
from theano.tensor import TensorType
from lasagne.init import Normal, HeNormal
from time import time
from PIL import Image
from matplotlib.pyplot import imshow, imsave, imread
from datasets import pix2pix
from lib.data_utils import processing_img, convert_img_back, convert_img, pix2pixBatch, shuffle_data, iter_data, ImgRescale

import numpy as np
import theano
import theano.tensor as T
import json
import lasagne
import h5py
import sys
import os


import models

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main():
    # Parameters
    train_data = './datasets/facades/train/'
    display_data = './datasets/facades/val/'
    start = 0
    stop = 400 
    save_samples = False
    shuffle_ = True 
    use_h5py = 0
    batchSize = 4 
    loadSize = 286
    fineSize = 256
    flip = True 
    ngf = 64
    ndf = 64 
    input_nc = 3
    output_nc = 3
    num_epoch = 1001
    training_method = 'adam'
    lr_G = 0.0002
    lr_D = 0.0002
    beta1 = 0.5

    task = 'facades'
    name = 'pan'
    which_direction = 'BtoA'
    preprocess = 'regular'
    begin_save = 700 
    save_freq = 100 
    show_freq = 20 
    continue_train = 0
    use_PercepGAN = 1
    use_Pix = 'No' 
    which_netG = 'unet_nodrop'
    which_netD = 'basic'
    lam_pix = 25.
    lam_p1 = 5.
    lam_p2 = 1.5
    lam_p3 = 1.5
    lam_p4 = 1.
    lam_gan_d = 1.
    lam_gan_g = 1.
    m = 3.0 
    test_deterministic = True   
    
    kD = 1
    kG = 1
    save_model_D = False
    # Load the dataset
    print("Loading data...")
    if which_direction == 'AtoB':
        tra_input, tra_output, _ = pix2pix(data_path=train_data, img_shape=[input_nc,loadSize,loadSize], save = save_samples, start=start, stop=stop)
        dis_input, dis_output, _ = pix2pix(data_path=display_data, img_shape=[input_nc,fineSize,fineSize], save = False, start=0, stop=4)
        dis_input = processing_img(dis_input, center=True, scale=True, convert=False)
    elif which_direction == 'BtoA':
        tra_output, tra_input, _ = pix2pix(data_path=train_data, img_shape=[input_nc,loadSize,loadSize], save = save_samples, start=start, stop=stop)
        dis_output, dis_input, _ = pix2pix(data_path=display_data, img_shape=[input_nc,fineSize,fineSize], save = False, start=0, stop=4)
        dis_input = processing_img(dis_input, center=True, scale=True, convert=False)
    ids = range(0,stop-start)  
    
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
    elif which_netG == 'unet_facades':
        generator = models.build_generator_facades(input_x,ngf=ngf)
    else:    
	print('waiting to fill')

    if use_PercepGAN == 1:
        if which_netD == 'basic':
            discriminator= models.build_discriminator(ndf=ndf)
        else:
	    print('waiting to fill')
    
    # Create expression for passing generator
    gen_imgs = lasagne.layers.get_output(generator)
   
    if use_PercepGAN == 1:
        # Create expression for passing real data through the discriminator
        dis1_f, dis2_f, dis3_f, dis4_f, disout_f = lasagne.layers.get_output(discriminator,input_y)
        # Create expression for passing fake data through the discriminator
        dis1_ff, dis2_ff, dis3_ff, dis4_ff, disout_ff = lasagne.layers.get_output(discriminator,gen_imgs)
        
	p1 = lam_p1*T.mean(T.abs_(dis1_f-dis1_ff)) 
	p2 = lam_p2*T.mean(T.abs_(dis2_f-dis2_ff)) 
	p3 = lam_p3*T.mean(T.abs_(dis3_f-dis3_ff)) 
	p4 = lam_p4*T.mean(T.abs_(dis4_f-dis4_ff)) 

        l2_norm = p1 + p2 + p3 + p4

	percepgan_dis_loss = lam_gan_d*(lasagne.objectives.binary_crossentropy(disout_f, 0.9) + lasagne.objectives.binary_crossentropy(disout_ff, 0)).mean() + T.maximum((T.constant(m)-l2_norm),T.constant(0.))
	percepgan_gen_loss = -lam_gan_g*(lasagne.objectives.binary_crossentropy(disout_ff, 0)).mean() + l2_norm
    else:
	l2_norm = T.constant(0) 
        percepgan_dis_loss = T.constant(0)
        percepgan_gen_loss = T.constant(0)
 
    if use_Pix == 'L1':
        pixel_loss = lam_pix*T.mean(abs(gen_imgs-input_y)) 
    elif use_Pix == 'L2':
        pixel_loss = lam_pix*T.mean(T.sqr(gen_imgs-input_y)) 
    else:
        pixel_loss = T.constant(0)

    # Create loss expressions
    generator_loss = percepgan_gen_loss + pixel_loss
    discriminator_loss = percepgan_dis_loss 
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    if training_method == 'adam': 
        g_updates = lasagne.updates.adam(
                generator_loss, generator_params, learning_rate=lr_G, beta1=beta1)
    elif training_method == 'nm':
        g_updates = lasagne.updates.nesterov_momentum(
                generator_loss, generator_params, learning_rate=lr_G, momentum=beta1)
 
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_g = theano.function([input_x, input_y], [p1, p2, p3, p4, l2_norm, generator_loss,pixel_loss], updates=g_updates)
    
    if use_PercepGAN == 1:
        discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
        if training_method == 'adam': 
	    d_updates = lasagne.updates.adam(
                    discriminator_loss, discriminator_params, learning_rate=lr_D, beta1=beta1)
        elif training_method == 'nm':
	    d_updates = lasagne.updates.nesterov_momentum(
                    discriminator_loss, discriminator_params, learning_rate=lr_D, momentum=beta1)
        train_d = theano.function([input_x, input_y], [l2_norm, discriminator_loss], updates=d_updates)
        dis_fn = theano.function([input_x, input_y], [(disout_f > .5).mean(), (disout_ff < .5).mean()])
    # Compile another function generating some data
    gen_fn = theano.function([input_x],
                             lasagne.layers.get_output(generator, deterministic=test_deterministic))
    
    # Finally, launch the training loop.
    print("Starting training...")
    
    desc = task + '_' + name 
    print desc
    
    f_log = open('logs/%s.ndjson'%desc, 'wb')
    log_fields = [
        'NE',
        'sec',
	'px',
	'1',
	'2',
	'3',
	'4',
	'pd',
        'cd',
	'pg',
        'cg',
        'fr',
        'tr',
    ]

    if not os.path.isdir('generated_imgs/'+desc):
        os.mkdir(os.path.join('generated_imgs/',desc))
    if not os.path.isdir('models/'+desc):
        os.mkdir(os.path.join('models/',desc))
       
    t = time()
    # We iterate over epochs:
    for epoch in range(num_epoch):
	if shuffle_ is True:
	    ids = shuffle_data(ids) 
        n_updates_g = 0
        n_updates_d = 0
        percep_d = 0
        percep_g = 0
        cost_g = 0
        cost_d = 0
	pixel = 0
        train_batches = 0
        k = 0
	p1 = 0
	p2 = 0
	p3 = 0
	p4 = 0
	for index_ in iter_data(ids, size=batchSize):
            index = sorted(index_) 
            xmb = tra_input[index,:,:,:]
            ymb = tra_output[index,:,:,:]
	    
	    if preprocess == 'regular':
                xmb, ymb = pix2pixBatch(xmb,ymb,fineSize,input_nc,flip=flip)
	    elif task == 'inpainting':
                print('waiting to fill')
	    elif task == 'cartoon':
                print('waiting to fill')
	    
	    if n_updates_g == 0: 
	        imsave('other/%s_input'%desc,convert_img_back(xmb[0,:,:,:]),format='png') 
	        imsave('other/%s_GT'%desc,convert_img_back(ymb[0,:,:,:]),format='png') 
	    
	    xmb = processing_img(xmb, center=True, scale=True, convert=False)
            ymb = processing_img(ymb, center=True, scale=True, convert=False)
            
	    if use_PercepGAN == 1:
		if k < kD:
		    percep, cost = train_d(xmb, ymb)
                    percep_d += percep 
                    cost_d += cost
                    n_updates_d += 1
                    k += 1
		elif k < kD + kG:
		    pp1, pp2, pp3, pp4, percep, cost, pix = train_g(xmb, ymb)
                    p1 += pp1
                    p2 += pp2
                    p3 += pp3
                    p4 += pp4
		    percep_g += percep 
                    cost_g += cost
                    pixel += pix 
                    n_updates_g += 1
                    k += 1
	        elif k == kD + kG:
		    percep, cost = train_d(xmb, ymb)
                    percep_d += percep 
                    cost_d += cost
                    n_updates_d += 1
		    pp1, pp2, pp3, pp4, percep, cost, pix = train_g(xmb, ymb)
                    p1 += pp1
                    p2 += pp2
                    p3 += pp3
                    p4 += pp4
                    percep_g += percep 
                    cost_g += cost
                    pixel += pix 
                    n_updates_g += 1
		if k == kD+kG:
		    k = 0
	    else: 
		pp1, pp2, pp3, pp4, percep, cost, pix = train_g(xmb, ymb)
                p1 += pp1
                p2 += pp2
                p3 += pp3
                p4 += pp4
                percep_g += percep 
                cost_g += cost
                pixel += pix 
                n_updates_g += 1
        
	if epoch%show_freq == 0:
            p1= p1/n_updates_g 
            p2= p2/n_updates_g 
            p3= p3/n_updates_g 
            p4= p4/n_updates_g 
            
	    percep_g = percep_g/n_updates_g 
            percep_d = percep_d/(n_updates_d + 0.0001)

            cost_g = cost_g/n_updates_g
            cost_d = cost_d/(n_updates_d + 0.0001)
            
            pixel = pixel/n_updates_g 
            
	    true_rate = -1
	    fake_rate = -1
            if use_PercepGAN == 1:
	        true_rate, fake_rate = dis_fn(xmb, ymb)

            log = [epoch, round(time()-t,2), round(pixel,2), round(p1,2), round(p2,2), round(p3,2), round(p4,2), round(percep_d,2), round(cost_d,2), round(percep_g,2), round(cost_g,2), round(float(fake_rate),2), round(float(true_rate),2)]
            print '%.0f %.2f %.2f %.2f %.2f %.2f% .2f %.2f %.2f %.2f% .2f %.2f'%(epoch, p1, p2, p3, p4, percep_d, cost_d, pixel, percep_g, cost_g, fake_rate, true_rate)
            
	    t = time()
            f_log.write(json.dumps(dict(zip(log_fields, log)))+'\n')
            f_log.flush()

            gen_imgs = gen_fn(dis_input)
            
	    blank_image = Image.new("RGB",(fineSize*4+5,fineSize*2+3))
	    pc=0
	    for i in range(2):
                for ii in range(4):
                    if i == 0:
		        img = dis_input[ii,:,:,:]
                        img = ImgRescale(img, center=True, scale=True, convert_back=True)
                        blank_image.paste(Image.fromarray(img),(ii*fineSize+ii+1,1)) 
                    elif i == 1:
                        img = gen_imgs[ii,:,:,:]
                        img = ImgRescale(img, center=True, scale=True, convert_back=True)
                        blank_image.paste(Image.fromarray(img),(ii*fineSize+ii+1,2+fineSize)) 
            blank_image.save('generated_imgs/%s/%s_%d.png'%(desc,desc,epoch))
	    
	    #pv = PatchViewer(grid_shape=(2, 4),
            #                 patch_shape=(256,256), is_color=True)
            #for i in range(2):
            #    for ii in range(4):
            #        if i == 0:
	    #            img = dis_input[ii,:,:,:]
            #        elif i == 1:
            #            img = gen_imgs[ii,:,:,:]
            #        img = convert_img_back(img)
            #        pv.add_patch(img, rescale=False, activation=0)
            
            #pv.save('generated_imgs/%s/%s_%d.png'%(desc,desc,epoch))

        if (epoch)%save_freq == 0 and epoch > begin_save - 1:
        # Optionally, you could now dump the network weights to a file like this:
            np.savez('models/%s/gen_%d.npz'%(desc,epoch), *lasagne.layers.get_all_param_values(generator))
            if use_PercepGAN == 1 and save_model_D is True:
                np.savez('models/%s/dis_%d.npz'%(desc,epoch), *lasagne.layers.get_all_param_values(discriminator))

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
