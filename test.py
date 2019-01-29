import scipy.io
import numpy
import matplotlib.pyplot as plt



row = 1500
column = 1024
size = row * column

file_path = 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/'
i_name = 512
aa = scipy.io.loadmat(file_path + 'slice_' + str(i_name).zfill(4) + '_truth.mat')['img_sino_truth']
print(aa.shape)

plt.figure()
plt.imshow(aa)
plt.colorbar()
plt.show()


def ConvNet_simple(self, input_tensor):
    '''
    This network is a six layer straightforward network.
    :param input_tensor:
    :return:
    '''
    PRINT = 1
    kernel_size = 3

    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x2 = tf.nn.relu(tf.layers.conv2d(x1, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x3 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x4 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x5 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x6 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x6 = tf.nn.relu(tf.layers.conv2d(x6, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    # output layer:
    output = tf.layers.conv2d(x6, 1, kernel_size=kernel_size, strides=(1, 1), padding='same')

    return output


def ConvNet_simple_residual(self, input_tensor):
    '''
    This network is a six layer straightforward network.
    :param input_tensor:
    :return:
    '''
    PRINT = 0
    kernel_size = 3

    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x2 = tf.nn.relu(tf.layers.conv2d(x1, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x3 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x4 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x5 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x6 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x6 = tf.nn.relu(tf.layers.conv2d(x6, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    # output layer:
    output = tf.layers.conv2d(x6, 1, kernel_size=kernel_size, strides=(1, 1), padding='same')

    return tf.math.add(output, input_tensor)


def ConvNet_12layers_residual(self, input_tensor):
    '''
    This network is a six layer straightforward network.
    :param input_tensor:
    :return:
    '''
    PRINT = 0
    kernel_size = 3

    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x2 = tf.nn.relu(tf.layers.conv2d(x1, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x3 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x4 = tf.nn.relu(tf.layers.conv2d(x3, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x5 = tf.nn.relu(tf.layers.conv2d(x4, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x6 = tf.nn.relu(tf.layers.conv2d(x5, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x6 = tf.nn.relu(tf.layers.conv2d(x6, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x7 = tf.nn.relu(tf.layers.conv2d(x6, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x7 = tf.nn.relu(tf.layers.conv2d(x7, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x8 = tf.nn.relu(tf.layers.conv2d(x7, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x8 = tf.nn.relu(tf.layers.conv2d(x8, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x9 = tf.nn.relu(tf.layers.conv2d(x8, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x9 = tf.nn.relu(tf.layers.conv2d(x9, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x10 = tf.nn.relu(tf.layers.conv2d(x9, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x10 = tf.nn.relu(tf.layers.conv2d(x10, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x11 = tf.nn.relu(tf.layers.conv2d(x10, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x11 = tf.nn.relu(tf.layers.conv2d(x11, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    x12 = tf.nn.relu(tf.layers.conv2d(x11, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))
    x12 = tf.nn.relu(tf.layers.conv2d(x12, 64, kernel_size=kernel_size, strides=(1, 1), padding='same'))

    # output layer:
    output = tf.layers.conv2d(x12, 1, kernel_size=kernel_size, strides=(1, 1), padding='same')

    return tf.math.add(output, input_tensor)


def EncoderDecoderTest(self, input_tensor):
    PRINT = 0

    kernel_size1 = 3
    kernel_size2 = 3
    strides = (1, 1, 1)

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.max_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), padding='same')
    print('Downsample')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.max_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), padding='same')
    print('Downsample')

    x3 = tf.reshape(x3, [-1, 128, 64])

    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.max_pooling1d(x3, pool_size=(2), strides=(2), padding='same')
    print('Downsample')

    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.max_pooling1d(x4, pool_size=(2), strides=(2), padding='same')
    print('Downsample')

    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x5.shape)

    # # do the upsampling:
    # y4
    y4 = tf.keras.layers.UpSampling1D().apply(x5)
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)

    # y3
    y3 = tf.keras.layers.UpSampling1D().apply(y4)
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)

    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    y2 = tf.keras.layers.UpSampling2D().apply(y3)
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)

    # y1
    y1 = tf.keras.layers.UpSampling2D().apply(y2)
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    # ## The last layer to output:
    # output = tf.layers.conv3d(y1, 1, kernel_size=4, strides=(1, 1, 1), padding='same')
    # if PRINT:
    #     print('Output shape: ', output.shape)
    return y1


def EncoderDecoderTest_residual(self, input_tensor):
    '''
    :param input_tensor:  the shape is: batch * depth * column * channel
    :return: the same shape as input_tensor
    '''
    PRINT = 0

    kernel_size1 = 3
    kernel_size2 = 3
    strides = (1, 1, 1)

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.max_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), padding='same')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.max_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), padding='same')

    x3 = tf.reshape(x3, [-1, 128, 64])

    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.max_pooling1d(x3, pool_size=(2), strides=(2), padding='same')

    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.max_pooling1d(x4, pool_size=(2), strides=(2), padding='same')

    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x5.shape)

    # # do the upsampling:
    # y4
    y4 = tf.keras.layers.UpSampling1D().apply(x5)
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)

    # y3
    y3 = tf.keras.layers.UpSampling1D().apply(y4)
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)

    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    y2 = tf.keras.layers.UpSampling2D().apply(y3)
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)

    # y1
    y1 = tf.keras.layers.UpSampling2D().apply(y2)
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    # ## The last layer to output:
    # output = tf.layers.conv3d(y1, 1, kernel_size=4, strides=(1, 1, 1), padding='same')
    # if PRINT:
    #     print('Output shape: ', output.shape)
    return tf.math.add(y1, input_tensor)


def EncoderDecoderTest_new(self, input_tensor):
    '''
    :param input_tensor:  the shape is: batch * depth * column * channel
    :return: the same shape as input_tensor
    '''
    PRINT = 0

    kernel_size1 = 3
    kernel_size2 = 3
    strides = (1, 1, 1)

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(2, 2), padding='same')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(2, 2), padding='same')

    x3 = tf.reshape(x3, [-1, 128, 64])

    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(2), padding='same')

    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(2), padding='same')

    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x5.shape)

    # # do the upsampling:
    # y4
    y4 = conv1d_transpose(x5, 256, kernel_size2, shape=(None, 32, 256))
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)

    # y3
    y3 = conv1d_transpose(y4, 128, kernel_size2, shape=(None, 64, 128))
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)

    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    y2 = tf.layers.conv2d_transpose(y3, filters=64, kernel_size=kernel_size2, strides=(2, 2), padding='same')
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)

    # y1
    y1 = tf.layers.conv2d_transpose(y2, filters=32, kernel_size=kernel_size2, strides=(2, 2), padding='same')
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    ## The last layer to output:
    output = tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same')
    if PRINT:
        print('Output shape: ', output.shape)
    return output


def EncoderDecoderTest_residual_new(self, input_tensor):
    '''
    :param input_tensor:  the shape is: batch * depth * column * channel
    :return: the same shape as input_tensor
    '''
    PRINT = 0

    kernel_size1 = 7
    kernel_size2 = 3

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(2, 2), padding='same')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(2, 2), padding='same')

    x3 = tf.reshape(x3, [-1, 128, 64])

    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(2), padding='same')

    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(2), padding='same')

    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x5.shape)

    # # do the upsampling:
    # y4
    y4 = conv1d_transpose(x5, 256, kernel_size2, shape=(None, 32, 256))
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)

    # y3
    y3 = conv1d_transpose(y4, 128, kernel_size2, shape=(None, 64, 128))
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)

    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    y2 = tf.layers.conv2d_transpose(y3, filters=64, kernel_size=kernel_size2, strides=(2, 2), padding='same')
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)

    # y1
    y1 = tf.layers.conv2d_transpose(y2, filters=32, kernel_size=kernel_size2, strides=(2, 2), padding='same')
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    ## The last layer to output:
    output = tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same')
    if PRINT:
        print('Output shape: ', output.shape)
    return tf.math.add(output, input_tensor)


def EncoderDecoderTest_residual_7depth(self, input_tensor):
    '''
    :param input_tensor:  the shape is: batch * depth * column * channel
    :return: the same shape as input_tensor
    '''
    PRINT = 0

    kernel_size1 = 3
    kernel_size2 = 3
    strides = (1, 1, 1)

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.max_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.max_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), padding='same')
    if PRINT:
        print('--Downsample--')

    x3 = tf.reshape(x3, [-1, 128, 64])
    if PRINT:
        print('--Reduce Dimension--')
    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.max_pooling1d(x3, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.max_pooling1d(x4, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x5_1d:
    if PRINT:
        print('X5_1d input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X5_1d output shape: ', x5.shape)
    x6 = tf.layers.max_pooling1d(x5, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x6_1d:
    if PRINT:
        print('X6_1d input shape: ', x6.shape)
    x6 = tf.nn.relu(tf.layers.conv1d(x6, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    x6 = tf.nn.relu(tf.layers.conv1d(x6, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X6_1d output shape: ', x6.shape)
    x7 = tf.layers.max_pooling1d(x6, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x7.shape)
    x7 = tf.nn.relu(tf.layers.conv1d(x7, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    x7 = tf.nn.relu(tf.layers.conv1d(x7, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x7.shape)
        print('--Upsample--')

    # # do the upsampling:
    # y6
    y6 = tf.keras.layers.UpSampling1D().apply(x7)
    if PRINT:
        print('Y6_1d input shape: ', y6.shape)
    y6 = tf.nn.relu(tf.layers.conv1d(y6, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    y6 = tf.nn.relu(tf.layers.conv1d(y6, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y6_1d output shape: ', y6.shape)
        print('--Upsample--')
    # y5
    y5 = tf.keras.layers.UpSampling1D().apply(y6)
    if PRINT:
        print('Y5_1d input shape: ', y5.shape)
    y5 = tf.nn.relu(tf.layers.conv1d(y5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    y5 = tf.nn.relu(tf.layers.conv1d(y5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y5_1d output shape: ', y5.shape)
        print('--Upsample--')
    # y4
    y4 = tf.keras.layers.UpSampling1D().apply(y5)
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)
        print('--Upsample--')
    # y3
    y3 = tf.keras.layers.UpSampling1D().apply(y4)
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)
        print('--Upsample--')
    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    if PRINT:
        print('--Expend Dimension--')

    y2 = tf.keras.layers.UpSampling2D().apply(y3)
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)
        print('--Upsample--')
    # y1
    y1 = tf.keras.layers.UpSampling2D().apply(y2)
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    ## The last layer to output:
    output = tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same')
    if PRINT:
        print('Output shape: ', output.shape)
    return tf.math.add(output, input_tensor)


def EncoderDecoderTest_7depth(self, input_tensor):
    '''
    :param input_tensor:  the shape is: batch * depth * column * channel
    :return: the same shape as input_tensor
    '''
    PRINT = 0

    kernel_size1 = 3
    kernel_size2 = 3
    strides = (1, 1, 1)

    ## head layer:
    if PRINT:
        print('Input shape: ', input_tensor.shape)
    x1 = tf.nn.relu(tf.layers.conv2d(input_tensor, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x1 = tf.nn.relu(tf.layers.conv2d(x1, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X1 output shape: ', x1.shape)
    x2 = tf.layers.max_pooling2d(x1, pool_size=(2, 2), strides=(2, 2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x2:
    if PRINT:
        print('X2 input shape: ', x2.shape)
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    x2 = tf.nn.relu(tf.layers.conv2d(x2, 64, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('X2 output shape: ', x2.shape)
    x3 = tf.layers.max_pooling2d(x2, pool_size=(2, 2), strides=(2, 2), padding='same')
    if PRINT:
        print('--Downsample--')

    x3 = tf.reshape(x3, [-1, 128, 64])
    if PRINT:
        print('--Reduce Dimension--')
    ## x3_1d:
    if PRINT:
        print('X3_1d input shape: ', x3.shape)
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    x3 = tf.nn.relu(tf.layers.conv1d(x3, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X3_1d output shape: ', x3.shape)
    x4 = tf.layers.max_pooling1d(x3, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x4_1d:
    if PRINT:
        print('X4_1d input shape: ', x4.shape)
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    x4 = tf.nn.relu(tf.layers.conv1d(x4, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X4_1d output shape: ', x4.shape)
    x5 = tf.layers.max_pooling1d(x4, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x5_1d:
    if PRINT:
        print('X5_1d input shape: ', x5.shape)
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    x5 = tf.nn.relu(tf.layers.conv1d(x5, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X5_1d output shape: ', x5.shape)
    x6 = tf.layers.max_pooling1d(x5, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## x6_1d:
    if PRINT:
        print('X6_1d input shape: ', x6.shape)
    x6 = tf.nn.relu(tf.layers.conv1d(x6, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    x6 = tf.nn.relu(tf.layers.conv1d(x6, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('X6_1d output shape: ', x6.shape)
    x7 = tf.layers.max_pooling1d(x6, pool_size=(2), strides=(2), padding='same')
    if PRINT:
        print('--Downsample--')
    ## the middle layer:
    if PRINT:
        print('middle input shape: ', x7.shape)
    x7 = tf.nn.relu(tf.layers.conv1d(x7, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    x7 = tf.nn.relu(tf.layers.conv1d(x7, 1024, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('middle output shape: ', x7.shape)
        print('--Upsample--')

    # # do the upsampling:
    # y6
    y6 = tf.keras.layers.UpSampling1D().apply(x7)
    if PRINT:
        print('Y6_1d input shape: ', y6.shape)
    y6 = tf.nn.relu(tf.layers.conv1d(y6, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    y6 = tf.nn.relu(tf.layers.conv1d(y6, 512, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y6_1d output shape: ', y6.shape)
        print('--Upsample--')
    # y5
    y5 = tf.keras.layers.UpSampling1D().apply(y6)
    if PRINT:
        print('Y5_1d input shape: ', y5.shape)
    y5 = tf.nn.relu(tf.layers.conv1d(y5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    y5 = tf.nn.relu(tf.layers.conv1d(y5, 256, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y5_1d output shape: ', y5.shape)
        print('--Upsample--')
    # y4
    y4 = tf.keras.layers.UpSampling1D().apply(y5)
    if PRINT:
        print('Y4_1d input shape: ', y4.shape)
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    y4 = tf.nn.relu(tf.layers.conv1d(y4, 128, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y4_1d output shape: ', y4.shape)
        print('--Upsample--')
    # y3
    y3 = tf.keras.layers.UpSampling1D().apply(y4)
    if PRINT:
        print('Y3_1d input shape: ', y3.shape)
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    y3 = tf.nn.relu(tf.layers.conv1d(y3, 64, kernel_size=kernel_size2, strides=(1), padding='same'))
    if PRINT:
        print('Y3_1d output shape: ', y3.shape)
        print('--Upsample--')
    # y2
    y3 = tf.reshape(y3, [-1, 1, 128, 64])
    if PRINT:
        print('--Expend Dimension--')

    y2 = tf.keras.layers.UpSampling2D().apply(y3)
    if PRINT:
        print('Y2 input shape: ', y2.shape)
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y2 = tf.nn.relu(tf.layers.conv2d(y2, 32, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y2 output shape: ', y2.shape)
        print('--Upsample--')
    # y1
    y1 = tf.keras.layers.UpSampling2D().apply(y2)
    if PRINT:
        print('Y1 input shape: ', y1.shape)
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    y1 = tf.nn.relu(tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same'))
    if PRINT:
        print('Y1 output shape: ', y1.shape)

    ## The last layer to output:
    output = tf.layers.conv2d(y1, 1, kernel_size=kernel_size1, strides=(1, 1), padding='same')
    # if PRINT:
    #     print('Output shape: ', output.shape)
    return output



