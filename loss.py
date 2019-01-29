import tensorflow as tf
import numpy as np
from scipy import signal
from basic_ops_p import *



class Loss():
    """
    :param y_pred: 2D tensor: (# batch, feature size) = (-1, 512*360*8)

    :param: y_truth: 2D tensor: (# batch, feature size) = (-1, 512*360*8)
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def loss(self, y_pred, y_truth, name = 'L2_norm'):
        img3d_pred = transform_to_3D_tf(y_pred)
        img3d_truth = transform_to_3D_tf(y_truth)

        if name == 'Huber_loss':
            l = self.Huber_loss(img3d_pred, img3d_truth)
        elif name == 'L2_norm':
            l = self.L2_norm(img3d_pred, img3d_truth)
        elif name == 'MS_SSIM_loss':
            l = 1 - self.tf_ms_ssim_3D(img3d_pred, img3d_truth)

        else:
            l = self.L2_norm(img3d_pred, img3d_truth)

        # # # **************************************************************************
        # # # Try different loss functions:
        # l = self.Huber_loss(img3d_pred, img3d_truth)

        # # # **************************************************************************
        # # # L2 norm loss
        # l = self.L2_norm(img3d_pred, img3d_truth)

        # # # **************************************************************************
        # # # MS_SSIM loss  % % the value is always NaN. Very wired.
        # l = 1 - self.tf_ms_ssim_3D(img5d_pred, img5d_truth)

        # # # **************************************************************************
        # # SSIM loss
        # l = 1 - self.tf_ssim_3D(img5d_pred, img5d_truth)
        # l = 1 - self.SSIM_3D(img5d_pred, img5d_truth)
        # # # **************************************************************************
        # # # L2 norm loss + MS_SSIM loss
        # l = self.L2_norm_3D(img5d_pred, img5d_truth) + self.tf_ms_ssim_3D(img5d_pred, img5d_truth)

        # # ***************************************************************************************************
        # # L2 norm loss + MS_SSIM loss + Perceptual loss
        # l = self.L2_norm_3D(img5d_pred, img5d_truth) + 2.0*(1 - self.tf_ms_ssim_3D(img5d_pred, img5d_truth)) \
        #     + 1.0*0.005*PerceptualLoss(self.batch_size).Perceptual_loss(img5d_pred[...,0,:], img5d_truth[...,0,:])

        # # ***************************************************************************************************
        # # L2 norm loss + SSIM loss
        # l = self.L2_norm_3D(img5d_pred, img5d_truth) + 0.5 * (1 - self.SSIM_3D_1batch(img5d_pred, img5d_truth))

        # # ***************************************************************************************************
        # # L2 norm loss + SSIM loss + Perceptual loss
        # l = self.L2_norm_3D(img5d_pred, img5d_truth) + 0.5 * (1 - self.SSIM_3D_1batch(img5d_pred, img5d_truth)) \
        #     + 0.2*0.005*PerceptualLoss(self.batch_size).Perceptual_loss(img5d_pred[...,0,:], img5d_truth[...,0,:])

        # l = self.RMSE(y_pred, y_truth)
        return l

    def Huber_loss(self, img1, img2):
        return tf.losses.huber_loss(img1, img2)

    def L2_norm(self, img1, img2):
        return tf.losses.mean_squared_error(labels=img1, predictions=img2)


    def SSIM_1batch(self, img1, img2):
        # pred & truth dimension: [#batch, size]
        batch_size = self.batch_size
        for j in range(batch_size):
            max_val = tf.maximum(tf.reduce_max(img1[j, ...]), tf.reduce_max(img2[j, ...]))
            min_val = tf.minimum(tf.reduce_min(img1[j, ...]), tf.reduce_min(img2[j, ...]))
            pred_nor = (img1[j, ...] - min_val) / (max_val - min_val)
            truth_nor = (img2[j, ...] - min_val) / (max_val - min_val)
            if j == 0:
                ssim = tf.image.ssim(img1=pred_nor, img2=truth_nor, max_val=1.0)
            else:
                ssim = ssim + tf.image.ssim(img1=pred_nor, img2=truth_nor, max_val=1.0)

        return tf.reduce_mean(ssim/batch_size)

    def RMSE(self, y_pred, y_truth):
        return tf.losses.mean_squared_error(labels=y_truth, predictions=y_pred)



    def SSIM(self, y_pred, y_truth, row=512, column=512):
        # pred & truth dimension: [#batch, size]
        batch_size = tf.shape(y_pred)[0]
        pred = transform_to_4D_tf(y_pred)
        truth = transform_to_4D_tf(y_truth)

        max_val = tf.maximum(tf.reduce_max(pred[...,0]), tf.reduce_max(truth[...,0]))
        min_val = tf.minimum(tf.reduce_min(pred[...,0]), tf.reduce_min(truth[...,0]))
        pred_nor = tf.manip.reshape((pred[...,0] - min_val) / (max_val - min_val), [batch_size, row, column, 1])
        truth_nor = tf.manip.reshape((truth[...,0] - min_val) / (max_val - min_val), [batch_size, row, column, 1])
        return self.tf_ms_ssim(pred_nor, truth_nor)

        #
        # ssim_arr = []
        # for i in range(8):
        #     max_val = tf.maximum(tf.reduce_max(pred[..., i]), tf.reduce_max(truth[..., i]))
        #     min_val = tf.minimum(tf.reduce_min(pred[..., i]), tf.reduce_min(truth[..., i]))
        #     pred_nor = tf.manip.reshape((pred[..., i] - min_val) / (max_val - min_val), [batch_size, row, column, 1])
        #     truth_nor = tf.manip.reshape((truth[..., i] - min_val) / (max_val - min_val), [batch_size, row, column, 1])
        #     ssim_arr.append(self.tf_ms_ssim(pred_nor, truth_nor, mean_metric=True, level=5))
        # return tf.reduce_mean(tf.stack(ssim_arr))

    def tf_ms_ssim_3D(self, img1, img2, weight = [1.0 for _ in range(4)], mean_metric=True, level=3):
        '''
        :param img1: 3D image with size of (batch, img_height, img_width, img_depth, channel)
        :param img2:
        :param weight:
        :param mean_metric:
        :param level:
        :return: multi-scale structure similarity index
        '''
        img_depth = img1.shape[3]
        weight = [w / sum(weight) for w in weight]
        for i in range(img_depth):
            if i == 0:
                loss = weight[i] * self.tf_ms_ssim(img1[...,i,:], img2[...,i,:], mean_metric=mean_metric, level=level)
            else:
                loss = tf.math.add(loss, weight[i] * self.tf_ms_ssim(img1[...,i,:], img2[...,i,:], mean_metric=mean_metric, level=level))
        return loss


    def tf_ms_ssim(self, img1, img2, mean_metric=True, level=3):
        '''
        :param img1: 4D tensor: NHWC
        :param img2: 4D tensor: NHWC
        :param mean_metric: return mean value or not
        :param level:
        :return:
        '''

        # # Adapted From Zhicheng's code
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = self.tf_ssim(img1, img2, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
                 (mssim[level - 1] ** weight[level - 1]))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

    def tf_ssim_3D(self, img1, img2, weight = [1.0 for _ in range(4)], cs_map=False, mean_metric=True, size=11, sigma=1.5):

        img_depth = img1.shape[3]
        weight = [w / sum(weight) for w in weight]
        for i in range(img_depth):
            if i == 0:
                loss = weight[i] * self.tf_ssim(img1[...,i,:], img2[...,i,:], cs_map=cs_map, mean_metric=mean_metric, size=size, sigma=sigma)[1]
            else:
                loss = tf.math.add(loss, weight[i] * self.tf_ssim(img1[...,i,:], img2[...,i,:], cs_map=cs_map, mean_metric=mean_metric, size=size, sigma=sigma)[1])
        return loss

    def tf_ssim(self, img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        '''
        :param img1: 4D tensor: NHWC
        :param img2: 4D tensor: NHWC
        :param cs_map:
        :param mean_metric:
        :param size:
        :param sigma:
        :return: ssim_map: ssim map;  mssim: mean value of ssim
        '''

        window = self._tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 0.0820  # tf.reduce_max(img1)  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                     (sigma1_sq + sigma2_sq + C2)),
                        (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))

        # if mean_metric:
        mssim = tf.reduce_mean(ssim_map)
        return ssim_map, mssim

    def _tf_fspecial_gauss(self, size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def ssim(self, img1, img2, cs_map=False):

        """ Adapted From Zhicheng's code
        Return the Structural Similarity Map corresponding to input images img1
        and img2 (images are assumed to be uint8)

        This function attempts to mimic precisely the functionality of ssim.m a
        MATLAB provided by the author's of SSIM
        https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
        """
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        size = 11
        sigma = 1.5
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        window = g / np.sum(g)
        # window = gauss.fspecial_gauss(size, sigma)
        K1 = 0.01
        K2 = 0.03
        L = 0.082  # bitdepth of image
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = signal.fftconvolve(window, img1, mode='valid')
        mu2 = signal.fftconvolve(window, img2, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
        if cs_map:
            ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                     (sigma1_sq + sigma2_sq + C2)),
                        (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    def total_generalized_variation_2(self, images):
        """
        Calculate the Total Variation for one or more images for use in denoising.

        https://en.wikipedia.org/wiki/Total_variation_denoising

        Args:
            images: `Tensor` with one or more images.
                    The shape is `[batch, height, width, channel]`.

        Returns:
            A scalar `Tensor` representing the value.
        """

        Ix = images[:, 1:, 1:, :] - images[:, :-1, 1:, :]
        Iy = images[:, 1:, 1:, :] - images[:, 1:, :-1, :]
        grad = tf.abs(Ix) + tf.abs(Iy)

        TgV = tf.div(tf.reduce_sum(tf.abs(grad[:, 1:, :, :] - grad[:, :-1, :, :])) + \
                     tf.reduce_sum(tf.abs(grad[:, :, 1:, :] - grad[:, :, :-1, :])), self.batch_size)

        # h_filter = tf.constant(np.array([[1,0,-1]]), dtype=tf.float32)
        # v_filter = tf.constant(np.array([[1],[0], [1]]), dtype=tf.float32)
        #
        # Ix = tf.nn.conv2d(images, h_filter, strides=[1,1,1,1], padding='VALID')
        # Iy = tf.nn.conv2d(images, v_filter, strides=[1,1,1,1], padding='VALID')

        return TgV

    def total_variation(self, images):
        """
        Calculate the Total Variation for one or more images for use in denoising.

        https://en.wikipedia.org/wiki/Total_variation_denoising

        Args:
            images: `Tensor` with one or more images.
                    The shape is `[batch, height, width, channel]`.

        Returns:
            A scalar `Tensor` representing the value.
        """

        value = tf.reduce_mean(tf.reduce_sum(tf.abs(images[:, 1:, :, :] - images[:, :-1, :, :]), reduction_indices=1) + \
                               tf.reduce_sum(tf.abs(images[:, :, 1:, :] - images[:, :, :-1, :]), reduction_indices=1))

        return value


class PerceptualLoss():

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_dict = np.load("vgg19.npy", encoding='latin1').item()

    def Perceptual_loss_3D(self, img1, img2, weight = [1.0 for _ in range(4)]):
        '''
        :param img1:  3D image with size of (batch, img_height, img_width, img_depth, channel)
        :param img2:  3D image
        :param name:
        :return:  perceptual loss
        '''
        img_depth = img1.shape[3]
        weight = [w / sum(weight) for w in weight]
        for i in range(img_depth):
            if i == 0:
                loss = weight[i] * self.Perceptual_loss(img1[...,i,:], img2[...,i,:])
            else:
                loss = tf.math.add(loss , weight[i] * self.Perceptual_loss(img1[...,i,:], img2[...,i,:]))
        return loss

    def Perceptual_loss(self, img1, img2, name=None):

        image1 = 255 * (img1[:, 16:-16, 16:-16, :] / 0.082)  # tf.image.resize_images(img1,[16:-16,16:-16])

        batch1 = tf.concat([image1, image1, image1], 3)  # img1

        # image2 = tf.image.resize_images(img2,[224,224])
        # # xmax, xmin = tf.reduce_max(image2), tf.reduce_min(image2)
        image2 = 255 * (img2[:, 16:-16, 16:-16, :] / 0.082)
        batch2 = tf.concat([image2, image2, image2], 3)  # img1

        Pimg1 = self.vgg19(batch1)
        Pimg2 = self.vgg19(batch2)
        # Pimg1 = vgg.build(batch1).conv3_4
        # Pimg2 = vgg.build(batch2).conv3_4
        # return tf.div(tf.reduce_sum(tf.square(tf.subtract(Pimg1,Pimg2))),batch)
        Pimg1 = 0.082 * tf.reshape(Pimg1, [self.batch_size, -1]) / 255.
        Pimg2 = 0.082 * tf.reshape(Pimg2, [self.batch_size, -1]) / 255.
        return tf.reduce_mean(tf.reduce_sum(tf.square(Pimg1 - Pimg2), reduction_indices=[1]))
        # tf.div(tf.reduce_sum(tf.square(tf.subtract(Pimg1,Pimg2))),batch)
        # tf.reduce_mean((tf.square(tf.subtract(Pimg1,Pimg2))))

    def vgg_avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def vgg_max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def vgg_conv_layer(self, img, name):
        with tf.variable_scope(name):
            filt = self.vgg_get_conv_filter(name)
            conv = tf.nn.conv2d(img, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.vgg_get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
        return relu

    def vgg_get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def vgg_get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def BN(self, img):
        batch_mean, batch_var = tf.nn.moments(img, [0, 1, 2], name='moments')
        img = tf.nn.batch_normalization(img, batch_mean, batch_var, 0, 1, 1e-3)
        return img

    def vgg19(self, img):
        with tf.variable_scope('vgg19', reuse=True):
            conv1_1 = self.vgg_conv_layer(img, "conv1_1")  # 224*224*64
            conv1_2 = self.vgg_conv_layer(conv1_1, "conv1_2")  # 224*224*64
            pool1 = self.vgg_max_pool(conv1_2, 'pool1')

            conv2_1 = self.vgg_conv_layer(pool1, "conv2_1")  # 112*112*128
            conv2_2 = self.vgg_conv_layer(conv2_1, "conv2_2")  # 112*112*128
            pool2 = self.vgg_max_pool(conv2_2, 'pool2')

            conv3_1 = self.vgg_conv_layer(pool2, "conv3_1")  # 56*56*256
            conv3_2 = self.vgg_conv_layer(conv3_1, "conv3_2")  # 56*56*256
            conv3_3 = self.vgg_conv_layer(conv3_2, "conv3_3")  # 56*56*256
            conv3_4 = self.vgg_conv_layer(conv3_3, "conv3_4")  # 56*56*256
            pool3 = self.vgg_max_pool(conv3_4, 'pool3')

            conv4_1 = self.vgg_conv_layer(pool3, "conv4_1")  # 28*28*512
            conv4_2 = self.vgg_conv_layer(conv4_1, "conv4_2")  # 28*28*512
            conv4_3 = self.vgg_conv_layer(conv4_2, "conv4_3")  # 28*28*512
            conv4_4 = self.vgg_conv_layer(conv4_3, "conv4_4")  # 28*28*512
            pool4 = self.vgg_max_pool(conv4_4, 'pool4')

            conv5_1 = self.vgg_conv_layer(pool4, "conv5_1")  # 14*14*512
            conv5_2 = self.vgg_conv_layer(conv5_1, "conv5_2")  # 14*14*512
            conv5_3 = self.vgg_conv_layer(conv5_2, "conv5_3")  # 14*14*512
            conv5_4 = self.vgg_conv_layer(conv5_3, "conv5_4")  # 14*14*512
            return conv5_4



