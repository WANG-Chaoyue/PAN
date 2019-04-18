from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.layers import Conv2DLayer
from lasagne.nonlinearities import LeakyRectify, sigmoid,tanh, softmax
from theano.tensor import TensorType
from lasagne.init import Normal, HeNormal

def build_generator_unet(input_var=None, ngf=64):
    # Input layer
    lrelu = LeakyRectify(0.2)
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1_ = Conv2DLayer(
            net, ngf*1, (3,3), (2,2), pad=1, W=Normal(0.05), nonlinearity=None)
    net1 = NonlinearityLayer(BatchNormLayer(net1_),nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2_ = Conv2DLayer(
            net1, ngf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net2 = NonlinearityLayer(BatchNormLayer(net2_),nonlinearity=lrelu)
    print ("Gen conv2:", net2.output_shape)
    net3_ = Conv2DLayer(
            net2, ngf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net3 = NonlinearityLayer(BatchNormLayer(net3_),nonlinearity=lrelu)
    print ("Gen conv3:", net3.output_shape)
    net4_ = Conv2DLayer(
            net3, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net4 = NonlinearityLayer(BatchNormLayer(net4_),nonlinearity=lrelu)
    print ("Gen conv4:", net4.output_shape)
    net5_ = Conv2DLayer(
            net4, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net5 = NonlinearityLayer(BatchNormLayer(net5_),nonlinearity=lrelu)
    print ("Gen conv5:", net5.output_shape)
    net6_ = Conv2DLayer(
            net5, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net6 = NonlinearityLayer(BatchNormLayer(net6_),nonlinearity=lrelu)
    print ("Gen conv6:", net6.output_shape)
    net7_ = Conv2DLayer(
            net6, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net7 = NonlinearityLayer(BatchNormLayer(net7_),nonlinearity=lrelu)
    print ("Gen conv7:", net7.output_shape)
    net8_ = Conv2DLayer(
            net7, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net8 = NonlinearityLayer(net8_)
    print ("Gen conv8:", net8.output_shape)
    # Decoder
    
    dnet1_ = DropoutLayer(BatchNormLayer(Deconv2DLayer(net8, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None)))
    skip1 = ConcatLayer([dnet1_,net7_])
    print ("skip layer 1:", skip1.output_shape)
    dnet1 = NonlinearityLayer(skip1)
    print ("Gen Deconv layer 1:", dnet1.output_shape)
    
    dnet2_ = DropoutLayer(BatchNormLayer(Deconv2DLayer(dnet1, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None)))
    skip2 = ConcatLayer([dnet2_,net6_])
    print ("skip layer 2:", skip2.output_shape)
    dnet2 = NonlinearityLayer(skip2)
    print ("Gen Deconv layer 2:", dnet2.output_shape)
   
    dnet3_ = DropoutLayer(BatchNormLayer(Deconv2DLayer(dnet2, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None)))
    skip3 = ConcatLayer([dnet3_,net5_])
    print ("skip layer 3:", skip3.output_shape)
    dnet3 = NonlinearityLayer(skip3)
    print ("Gen Deconv layer 3:", dnet3.output_shape)
    
    dnet4_ = BatchNormLayer(Deconv2DLayer(dnet3, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip4 = ConcatLayer([dnet4_,net4_])
    print ("skip layer 4:", skip4.output_shape)
    dnet4 = NonlinearityLayer(skip4)
    print ("Geneartor deconv 4:", dnet4.output_shape)
    
    dnet5_ = BatchNormLayer(Deconv2DLayer(dnet4, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip5 = ConcatLayer([dnet5_,net3_])
    print ("skip layer 5:", skip5.output_shape)
    dnet5 = NonlinearityLayer(skip5)
    print ("Geneartor deconv 5:", dnet5.output_shape)
    
    dnet6_ = BatchNormLayer(Deconv2DLayer(dnet5, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip6 = ConcatLayer([dnet6_,net2_])
    print ("skip layer 6:", skip6.output_shape)
    dnet6 = NonlinearityLayer(skip6)
    print ("Geneartor deconv 6:", dnet6.output_shape)
    
    dnet7_ = BatchNormLayer(Deconv2DLayer(dnet6, ngf, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip7 = ConcatLayer([dnet7_,net1_])
    print ("skip layer 7:", skip7.output_shape)
    dnet7 = NonlinearityLayer(skip7)
    print ("Geneartor deconv 7:", dnet7.output_shape)
    
    dnet_out = Deconv2DLayer(dnet7,3,(4,4),(2,2),crop=1,W=Normal(0.02),nonlinearity=tanh)
    
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out

def build_generator_unet_nodrop(input_var=None, ngf=64):
    # Input layer
    lrelu = LeakyRectify(0.2)
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1_ = Conv2DLayer(
            net, ngf*1, (3,3), (2,2), pad=1, W=Normal(0.05), nonlinearity=None)
    net1 = NonlinearityLayer(BatchNormLayer(net1_),nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2_ = Conv2DLayer(
            net1, ngf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net2 = NonlinearityLayer(BatchNormLayer(net2_),nonlinearity=lrelu)
    print ("Gen conv2:", net2.output_shape)
    net3_ = Conv2DLayer(
            net2, ngf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net3 = NonlinearityLayer(BatchNormLayer(net3_),nonlinearity=lrelu)
    print ("Gen conv3:", net3.output_shape)
    net4_ = Conv2DLayer(
            net3, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net4 = NonlinearityLayer(BatchNormLayer(net4_),nonlinearity=lrelu)
    print ("Gen conv4:", net4.output_shape)
    net5_ = Conv2DLayer(
            net4, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net5 = NonlinearityLayer(BatchNormLayer(net5_),nonlinearity=lrelu)
    print ("Gen conv5:", net5.output_shape)
    net6_ = Conv2DLayer(
            net5, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net6 = NonlinearityLayer(BatchNormLayer(net6_),nonlinearity=lrelu)
    print ("Gen conv6:", net6.output_shape)
    net7_ = Conv2DLayer(
            net6, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net7 = NonlinearityLayer(BatchNormLayer(net7_),nonlinearity=lrelu)
    print ("Gen conv7:", net7.output_shape)
    net8_ = Conv2DLayer(
            net7, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net8 = NonlinearityLayer(net8_)
    print ("Gen conv8:", net8.output_shape)
    # Decoder
    
    dnet1_ = BatchNormLayer(Deconv2DLayer(net8, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip1 = ConcatLayer([dnet1_,net7_])
    print ("skip layer 1:", skip1.output_shape)
    dnet1 = NonlinearityLayer(skip1)
    print ("Gen Deconv layer 1:", dnet1.output_shape)
    
    dnet2_ = BatchNormLayer(Deconv2DLayer(dnet1, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip2 = ConcatLayer([dnet2_,net6_])
    print ("skip layer 2:", skip2.output_shape)
    dnet2 = NonlinearityLayer(skip2)
    print ("Gen Deconv layer 2:", dnet2.output_shape)
   
    dnet3_ = BatchNormLayer(Deconv2DLayer(dnet2, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip3 = ConcatLayer([dnet3_,net5_])
    print ("skip layer 3:", skip3.output_shape)
    dnet3 = NonlinearityLayer(skip3)
    print ("Gen Deconv layer 3:", dnet3.output_shape)
    
    dnet4_ = BatchNormLayer(Deconv2DLayer(dnet3, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip4 = ConcatLayer([dnet4_,net4_])
    print ("skip layer 4:", skip4.output_shape)
    dnet4 = NonlinearityLayer(skip4)
    print ("Geneartor deconv 4:", dnet4.output_shape)
    
    dnet5_ = BatchNormLayer(Deconv2DLayer(dnet4, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip5 = ConcatLayer([dnet5_,net3_])
    print ("skip layer 5:", skip5.output_shape)
    dnet5 = NonlinearityLayer(skip5)
    print ("Geneartor deconv 5:", dnet5.output_shape)
    
    dnet6_ = BatchNormLayer(Deconv2DLayer(dnet5, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip6 = ConcatLayer([dnet6_,net2_])
    print ("skip layer 6:", skip6.output_shape)
    dnet6 = NonlinearityLayer(skip6)
    print ("Geneartor deconv 6:", dnet6.output_shape)
    
    dnet7_ = BatchNormLayer(Deconv2DLayer(dnet6, ngf, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip7 = ConcatLayer([dnet7_,net1_])
    print ("skip layer 7:", skip7.output_shape)
    dnet7 = NonlinearityLayer(skip7)
    print ("Geneartor deconv 7:", dnet7.output_shape)
    
    dnet_out = Deconv2DLayer(dnet7,3,(4,4),(2,2),crop=1,W=Normal(0.02),nonlinearity=tanh)
    
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out

def build_generator_unet_1(input_var=None, ngf=64):
    # Input layer
    lrelu = LeakyRectify(0.2)
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1_ = Conv2DLayer(
            net, ngf*1, (3,3), (2,2), pad=1, W=Normal(0.05), nonlinearity=None)
    net1 = NonlinearityLayer(BatchNormLayer(net1_),nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2_ = Conv2DLayer(
            net1, ngf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net2 = NonlinearityLayer(BatchNormLayer(net2_),nonlinearity=lrelu)
    print ("Gen conv2:", net2.output_shape)
    net3_ = Conv2DLayer(
            net2, ngf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net3 = NonlinearityLayer(BatchNormLayer(net3_),nonlinearity=lrelu)
    print ("Gen conv3:", net3.output_shape)
    net4_ = Conv2DLayer(
            net3, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net4 = NonlinearityLayer(BatchNormLayer(net4_),nonlinearity=lrelu)
    print ("Gen conv4:", net4.output_shape)
    net5_ = Conv2DLayer(
            net4, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net5 = NonlinearityLayer(BatchNormLayer(net5_),nonlinearity=lrelu)
    print ("Gen conv5:", net5.output_shape)
    net6_ = Conv2DLayer(
            net5, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net6 = NonlinearityLayer(BatchNormLayer(net6_),nonlinearity=lrelu)
    print ("Gen conv6:", net6.output_shape)
    # Decoder
    
    dnet3_ = BatchNormLayer(Deconv2DLayer(net6, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip3 = ConcatLayer([dnet3_,net5_])
    print ("skip layer 3:", skip3.output_shape)
    dnet3 = NonlinearityLayer(skip3)
    print ("Gen Deconv layer 3:", dnet3.output_shape)
    
    dnet4_ = BatchNormLayer(Deconv2DLayer(dnet3, ngf*8, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip4 = ConcatLayer([dnet4_,net4_])
    print ("skip layer 4:", skip4.output_shape)
    dnet4 = NonlinearityLayer(skip4)
    print ("Geneartor deconv 4:", dnet4.output_shape)
    
    dnet5_ = BatchNormLayer(Deconv2DLayer(dnet4, ngf*4, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip5 = ConcatLayer([dnet5_,net3_])
    print ("skip layer 5:", skip5.output_shape)
    dnet5 = NonlinearityLayer(skip5)
    print ("Geneartor deconv 5:", dnet5.output_shape)
    
    dnet6_ = BatchNormLayer(Deconv2DLayer(dnet5, ngf*2, (4,4), (2,2), crop=1, W=Normal(0.02),nonlinearity=None))
    skip6 = ConcatLayer([dnet6_,net2_])
    print ("skip layer 6:", skip6.output_shape)
    dnet6 = NonlinearityLayer(skip6)
    print ("Geneartor deconv 6:", dnet6.output_shape)
    
    dnet7 = batch_norm(Deconv2DLayer(dnet6, ngf, (4,4), (2,2), crop=1, W=Normal(0.02)))
    print ("Geneartor deconv 7:", dnet7.output_shape)
    
    dnet_out = Deconv2DLayer(dnet7,3,(4,4),(2,2),crop=1,W=Normal(0.02),nonlinearity=tanh)
    
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out

def build_generator_deraining(input_var=None, ngf=64):
    lrelu = LeakyRectify(0.2)
    # Input layer
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1 = Conv2DLayer(net, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2_ = BatchNormLayer(Conv2DLayer(net1, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=None))
    net2  = NonlinearityLayer(net2_,nonlinearity=lrelu)
    print ("Gen conv2:", net2.output_shape)
    net3 = batch_norm(Conv2DLayer(net2, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv3:", net3.output_shape)
    net4_ = BatchNormLayer(Conv2DLayer(net3, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=None))
    net4  = NonlinearityLayer(net4_,nonlinearity=lrelu)
    print ("Gen conv4:", net4.output_shape)
    net5 = batch_norm(Conv2DLayer(net4, 32, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv5:", net5.output_shape)
    net6 = batch_norm(Conv2DLayer(net5, 1, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv6:", net6.output_shape)
    # Decoder
    dnet1 = batch_norm(Deconv2DLayer(net6, 32, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 1:", dnet1.output_shape)
    dnet2_ = BatchNormLayer(Deconv2DLayer(dnet1, ngf, (3,3), (1,1), crop=1, W=Normal(0.05),nonlinearity=None))
    skip1 = ElemwiseSumLayer([dnet2_, net4_])
    dnet2 = NonlinearityLayer(skip1)
    print ("Gen Deconv layer 2:", dnet2.output_shape)
    dnet3 = batch_norm(Deconv2DLayer(dnet2, ngf, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 3:", dnet3.output_shape)
    dnet4_ = BatchNormLayer(Deconv2DLayer(dnet3, ngf, (3,3), (1,1), crop=1, W=Normal(0.05),nonlinearity=None))
    skip2 = ElemwiseSumLayer([dnet4_, net2_])
    dnet4 = NonlinearityLayer(skip2)
    print ("Gen Deconv layer 4:", dnet4.output_shape)
    dnet5 = batch_norm(Deconv2DLayer(dnet4, ngf, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 5:", dnet5.output_shape)
    dnet_out = Deconv2DLayer(dnet5,3,(3,3),(1,1),crop=1,W=Normal(0.05),nonlinearity=tanh)
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out

def build_generator_facades(input_var=None, ngf=64):
    lrelu = LeakyRectify(0.2)
    # Input layer
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1 = Conv2DLayer(net, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2_ = BatchNormLayer(Conv2DLayer(net1, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=None))
    net2  = NonlinearityLayer(net2_,nonlinearity=lrelu)
    print ("Gen conv2:", net2.output_shape)
    net3 = batch_norm(Conv2DLayer(net2, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv3:", net3.output_shape)
    net4_ = BatchNormLayer(Conv2DLayer(net3, ngf, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=None))
    net4  = NonlinearityLayer(net4_,nonlinearity=lrelu)
    print ("Gen conv4:", net4.output_shape)
    net5 = batch_norm(Conv2DLayer(net4, 32, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv5:", net5.output_shape)
    net6 = batch_norm(Conv2DLayer(net5, 1, (3,3), (1,1), pad=1, W=Normal(0.05),nonlinearity=lrelu))
    print ("Gen conv6:", net6.output_shape)
    # Decoder
    dnet1 = batch_norm(Deconv2DLayer(net6, 32, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 1:", dnet1.output_shape)
    dnet2_ = BatchNormLayer(Deconv2DLayer(dnet1, ngf, (3,3), (1,1), crop=1, W=Normal(0.05),nonlinearity=None))
    skip1 = ElemwiseSumLayer([dnet2_, net4_])
    dnet2 = NonlinearityLayer(skip1)
    print ("Gen Deconv layer 2:", dnet2.output_shape)
    dnet3 = batch_norm(Deconv2DLayer(dnet2, ngf, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 3:", dnet3.output_shape)
    dnet4_ = BatchNormLayer(Deconv2DLayer(dnet3, ngf, (3,3), (1,1), crop=1, W=Normal(0.05),nonlinearity=None))
    skip2 = ElemwiseSumLayer([dnet4_, net2_])
    dnet4 = NonlinearityLayer(skip2)
    print ("Gen Deconv layer 4:", dnet4.output_shape)
    dnet5 = batch_norm(Deconv2DLayer(dnet4, ngf, (3,3), (1,1), crop=1, W=Normal(0.05)))
    print ("Gen Deconv layer 5:", dnet5.output_shape)
    dnet_out = Deconv2DLayer(dnet5,3,(3,3),(1,1),crop=1,W=Normal(0.05),nonlinearity=tanh)
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out


def build_generator_inpainting(input_var=None, ngf=64):
    # Input layer
    lrelu = LeakyRectify(0.2)
    net = InputLayer(shape=(None, 3, 128, 128), input_var=input_var)
    print ("Generator input:", net.output_shape)
    # ConvLayer
    net1 = Conv2DLayer(
            net, ngf, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu)
    print ("Gen conv1:", net1.output_shape)
    net2 = batch_norm(Conv2DLayer(
            net1, ngf, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Gen conv2:", net2.output_shape)
    net3 = batch_norm(Conv2DLayer(
            net2, ngf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Gen conv3:", net3.output_shape)
    net4 = batch_norm(Conv2DLayer(
            net3, ngf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Gen conv4:", net4.output_shape)
    net5 = batch_norm(Conv2DLayer(
            net4, ngf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Gen conv5:", net5.output_shape)
    net6 = batch_norm(Conv2DLayer(
            net5, 4000, (4,4), pad=0, W=Normal(0.02), nonlinearity=lrelu))
    print ("Gen conv6:", net6.output_shape)
    # Decoder
    dnet1 = batch_norm(Deconv2DLayer(net6,ngf*8,(4,4),(1,1),crop=0,W=Normal(0.02)))
    print ("Gen Deconv layer 1:", dnet1.output_shape)
    
    dnet2 = batch_norm(Deconv2DLayer(dnet1,ngf*4,(4,4),(2,2),crop=1,W=Normal(0.02)))
    print ("Gen Deconv layer 2:", dnet2.output_shape)
    
    dnet3 = batch_norm(Deconv2DLayer(dnet2,ngf*2,(4,4),(2,2),crop=1,W=Normal(0.02)))
    print ("Gen Deconv layer 3:", dnet3.output_shape)

    dnet4 = batch_norm(Deconv2DLayer(dnet3,ngf,(4,4),(2,2),crop=1,W=Normal(0.02)))
    print ("Geneartor deconv 4:", dnet4.output_shape)
    
    dnet_out = Deconv2DLayer(dnet4,3,(4,4),(2,2),crop=1,W=Normal(0.02),nonlinearity=tanh)
    
    print ("Generator output:", dnet_out.output_shape)
    print (' ')
    return dnet_out

def build_discriminator(input_var=None, ndf=64):
    lrelu = LeakyRectify(0.2)
    # input: true images
    net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
    print ("Discriminator input:", net.output_shape)
    net1_ = Conv2DLayer(
            net, ndf, (3,3), (1,1), pad=1, W=Normal(0.5), nonlinearity=None)
    net1 = NonlinearityLayer(net1_,nonlinearity=tanh)
    print ("Discriminator conv1:", net1.output_shape)
    net2 = batch_norm(Conv2DLayer(
            net1, ndf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Discriminator conv2:", net2.output_shape)
    net2 = batch_norm(Conv2DLayer(
            net2, ndf*2, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net2_ = Conv2DLayer(
            net2, ndf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net2 = NonlinearityLayer(BatchNormLayer(net2_),nonlinearity=lrelu)
    print ("Discriminator conv2:", net2.output_shape)
    net3 = batch_norm(Conv2DLayer(
            net2, ndf*4, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net3_ = Conv2DLayer(
            net3, ndf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net3 = NonlinearityLayer(BatchNormLayer(net3_),nonlinearity=lrelu)
    print ("Discriminator conv3:", net3.output_shape)
    net4 = batch_norm(Conv2DLayer(
            net3, ndf*8, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net4_ = Conv2DLayer(
	    net4, ndf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net4 = NonlinearityLayer(BatchNormLayer(net4_),nonlinearity=lrelu)
    print ("Discriminator conv4:", net4.output_shape)
    net5 = Conv2DLayer(
            net4, 8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    print ("Discriminator conv5:", net5.output_shape)
    net_out = DenseLayer(net5, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Discriminator output:", net_out.output_shape)
    print (' ')
    return net1, net2, net3, net4, net_out

def build_discriminator_inpainting(input_var=None, ndf=64):
    lrelu = LeakyRectify(0.2)
    # input: true images
    net = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    print ("Discriminator input:", net.output_shape)
    net1_ = Conv2DLayer(
            net, ndf, (3,3), (1,1), pad=0, W=Normal(0.5), nonlinearity=None)
    net1 = NonlinearityLayer(net1_,nonlinearity=tanh)
    print ("Discriminator conv1:", net1.output_shape)
    net2 = batch_norm(Conv2DLayer(
            net1, ndf*2, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    print ("Discriminator conv2:", net2.output_shape)
    net2 = batch_norm(Conv2DLayer(
            net2, ndf*2, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net2_ = Conv2DLayer(
            net2, ndf*4, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net2 = NonlinearityLayer(BatchNormLayer(net2_),nonlinearity=lrelu)
    print ("Discriminator conv2:", net2.output_shape)
    net3 = batch_norm(Conv2DLayer(
            net2, ndf*4, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net3_ = Conv2DLayer(
            net3, ndf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net3 = NonlinearityLayer(BatchNormLayer(net3_),nonlinearity=lrelu)
    print ("Discriminator conv3:", net3.output_shape)
    net4 = batch_norm(Conv2DLayer(
            net3, ndf*8, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=lrelu))
    net4_ = Conv2DLayer(
	    net4, ndf*8, (3,3), (2,2), pad=1, W=Normal(0.02), nonlinearity=None)
    net4 = NonlinearityLayer(BatchNormLayer(net4_),nonlinearity=lrelu)
    print ("Discriminator conv4:", net4.output_shape)
    net5 = Conv2DLayer(
            net4, 8, (3,3), (1,1), pad=1, W=Normal(0.02), nonlinearity=None)
    print ("Discriminator conv5:", net5.output_shape)
    net_out = DenseLayer(net5, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print ("Discriminator output:", net_out.output_shape)
    print (' ')
    return net1, net2, net3, net4, net_out
