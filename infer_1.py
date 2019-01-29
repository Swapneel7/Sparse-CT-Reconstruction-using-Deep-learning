import scipy.io
import time
import os
import matplotlib.pyplot as plt
from basic_ops import *
from network import *
from evaluation import *
from basic_ops_p import *

FLAG = Flag()
tf.reset_default_graph()
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# The file has to be rank 2.
# And in diemension of view * det_pixel
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# tunning parametering
TRAIN_NAME = 'train0118_06'
NET_NAME = 'UNet_Res_4layers'
# RESTORED_WEIGHT = "C:/Projects/Sparse_View/weights/" + TRAIN_NAME + "/unet3D.ckpt-" + str(40)
RESTORED_WEIGHT = "C:/Projects/Sparse_View/weights/" + TRAIN_NAME + "/best_rmse.ckpt-" + str(77)

DATA_FOLDER = 'C:/Projects/Sparse_View_data/sinogram_data/scan_7/'
IMG_NAME = '0512'
FLAG.SAVE_DATA = True



# # fixed parameters
PATCH_ROW = 64
PATCH_COL = 64
BATCH_SIZE = 1

# # dependent variables:
OUT_NAME = TRAIN_NAME
FILE_TEST = DATA_FOLDER + 'slice_' + IMG_NAME + '_inter.mat'
FILE_TRUTH = DATA_FOLDER + 'slice_' + IMG_NAME + '_truth.mat'
FILE_OUT = 'C:/Projects/Sparse_View_data/inference/' + OUT_NAME + '/'
SIZE = PATCH_ROW*PATCH_COL



if __name__ == '__main__':

    #########################################################################
    # # read data:
    feature_img = scipy.io.loadmat(FILE_TEST)['img_sino_inter'].astype(np.float32)
    label_img = scipy.io.loadmat(FILE_TRUTH)['img_sino_truth'].astype(np.float32)
    feature_img = np.nan_to_num(feature_img)
    label_img = np.nan_to_num(label_img)
    # feature_img[np.logical_or(feature_img > 500, feature_img < -500)] = 0.
    # label_img[np.logical_or(label_img > 500, label_img < -500)] = 0.

    ROW = np.shape(feature_img)[0]
    COL = np.shape(feature_img)[1]
    NUM_PATCH_ROW = int(np.ceil(float(ROW)/PATCH_ROW))
    NUM_PATCH_COL = int(np.ceil(float(COL) / PATCH_COL))
    #########################################################################


    #########################################################################
    # # load the data:
    with tf.name_scope('inputs'):
        input_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, SIZE], name='feature')
        label_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, SIZE], name='label')
    #########################################################################


    #########################################################################
    # #  network
    model = Network().model
    pred_tensor = model(x_input=input_tensor, name = NET_NAME)
    #########################################################################


    #########################################################################
    # # Evaluation Metrics
    with tf.name_scope('eval'):
        RMSE_eval_before, up_op_rmse_bef = Evaluation(BATCH_SIZE).RMSE(y_pred=input_tensor, y_truth=label_tensor)
        RMSE_eval_after, up_op_rmse_aft = Evaluation(BATCH_SIZE).RMSE(y_pred=pred_tensor, y_truth=label_tensor)
        SSIM_eval_before = Evaluation(BATCH_SIZE).SSIM(y_pred=input_tensor, y_truth=label_tensor)
        SSIM_eval_after = Evaluation(BATCH_SIZE).SSIM(y_pred=pred_tensor, y_truth=label_tensor)
    #########################################################################


    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, RESTORED_WEIGHT)
        print('Testing Starts Now! ...............')
        start = time.clock()
        pred_img = np.zeros([ROW, COL], dtype=feature_img.dtype)
        for i_row in range(NUM_PATCH_ROW):
            range_row = list(range(i_row * PATCH_ROW, min((i_row + 1) * PATCH_ROW, ROW)))
            for i_col in range(NUM_PATCH_COL):
                range_col = list(range(i_col * PATCH_COL, min((i_col + 1) * PATCH_COL, COL)))

                feature_patch = np.zeros([PATCH_ROW, PATCH_COL])
                label_patch = np.zeros([PATCH_ROW, PATCH_COL])
                feature_patch[0:len(range_row), 0:len(range_col)] = feature_img[i_row * PATCH_ROW : min((i_row + 1) * PATCH_ROW, ROW),
                                                                                i_col * PATCH_COL : min((i_col + 1) * PATCH_COL, COL)]
                label_patch[0:len(range_row), 0:len(range_col)] = label_img[i_row * PATCH_ROW : min((i_row + 1) * PATCH_ROW, ROW),
                                                                            i_col * PATCH_COL : min((i_col + 1) * PATCH_COL, COL)]
                feature_patch = transform_to_2D_np(feature_patch)
                label_patch = transform_to_2D_np(label_patch)
                # run the graph:
                pred_img_patch, _,  _= sess.run([pred_tensor, up_op_rmse_bef, up_op_rmse_aft],
                                                       feed_dict={input_tensor: feature_patch, label_tensor: label_patch})

                pred_img_patch = np.squeeze(transform_to_3D_np(pred_img_patch))
                pred_img[i_row * PATCH_ROW : min((i_row + 1) * PATCH_ROW, ROW),
                                                                            i_col * PATCH_COL : min((i_col + 1) * PATCH_COL, COL)] \
                    = pred_img_patch[0:len(range_row), 0:len(range_col)]

        elapsed = (time.clock() - start)
        print("Time used: ", elapsed)
        print("RMSE: Before correction: {:.7f}...)".format(RMSE_eval_before.eval()))
        print("RMSE: After  correction: {:.7f}...)".format(RMSE_eval_after.eval()))

    # # save the images out:
    if FLAG.SAVE_DATA:
        if not os.path.exists(FILE_OUT):
            os.makedirs(FILE_OUT)
        feature_img.astype('float32').tofile(FILE_OUT + 'img_' + IMG_NAME + '_inter.raw')
        pred_img.astype('float32').tofile(FILE_OUT + 'img_' + IMG_NAME + '_correct.raw')
        label_img.astype('float32').tofile(FILE_OUT + 'img_' + IMG_NAME + '_truth.raw')
        # # save to mat files to the mat folder:
        # scipy.io.savemat(FILE_MAT + IMG_NAME + '_sino_log_real_mean_proc.mat', mdict={'proj_attlog_real_mean_proc': np.transpose(pred_img, [2,0,1])})
    #######################################################################
    #  # Show the image:

    contrast_display1 = (np.min(label_img[:, 80:COL-80]), np.max(label_img[:, 80:COL-80]))
    contrast_display2 = ( -np.max(np.abs( feature_img[:, 80:COL-80] - label_img[:, 80:COL-80])),
                           np.max(np.abs( feature_img[:, 80:COL-80] - label_img[:, 80:COL-80])))

    contrast_display1 = (np.min(label_img[:, 250:COL-250]), np.max(label_img[:, 250:COL-250]))
    contrast_display2 = ( -np.max(np.abs( feature_img[:, 250:COL-250] - label_img[:, 250:COL-250])),
                           np.max(np.abs( feature_img[:, 250:COL-250] - label_img[:, 250:COL-250])))

    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(feature_img), cmap='gray', vmin=contrast_display1[0], vmax=contrast_display1[1])
    plt.title('Interpolation')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.squeeze(pred_img), cmap='gray', vmin=contrast_display1[0], vmax=contrast_display1[1])
    plt.title('Correction')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.squeeze(label_img), cmap='gray', vmin=contrast_display1[0], vmax=contrast_display1[1])
    plt.title('Truth')
    plt.colorbar()


    plt.figure()
    plt.subplot(121)
    plt.imshow(np.squeeze(feature_img - label_img), cmap='jet', vmin=contrast_display2[0], vmax=contrast_display2[1])
    plt.colorbar()
    plt.title('Diff Image (Before Correction)')
    plt.subplot(122)
    plt.imshow(np.squeeze(pred_img - label_img), cmap='jet', vmin=contrast_display2[0], vmax=contrast_display2[1])
    plt.colorbar()
    plt.title('Diff Image (After Correction)')
    plt.show()


    print('END! .................')

