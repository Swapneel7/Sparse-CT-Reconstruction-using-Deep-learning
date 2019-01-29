from TF_data_gene import *
from loss import *
from network import *
from evaluation import *
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAG = Flag()

# tunning parametering
train_name = 'test0121_09'
NET_NAME = 'UNet_Res_4layers'  # # optins: Conv_Res_Nlayers, EncoDeco_Res_Nlayers, UNet_Res_Nlayers,
LOSS_NAME = 'L2_norm'

EPOCH = 1000
TRAIN_SIZE = 400
VALID_SIZE = 400
FILE_TRAIN = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p400_size64x64_test400.tfrecords'
FILE_VALID = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p400_size64x64_test400.tfrecords'
LR = 0.001
LR_DECAY_STEP = 100
LR_DECAY_RATE = 0.95
FLAG.SAVEWEIGHT = True
OPTIMIZER = 'ADAM'   # choose from: 'SGD', 'ADAM'

# restore weight
FLAG.RESTORE_GRAPH = True
RESTORED_WEIGHT = "C:/Projects/Sparse_View/weights/" + "test0121_08" + "/unet3D.ckpt-" + str(1000)


# # fixed parameters
ROW = 64
COLUMN = 64
BATCH_SIZE = 400

# # dependent variables:
SIZE = ROW*COLUMN

if FLAG.RESTORE_GRAPH:
    tf.reset_default_graph()
if __name__ == '__main__':
    save_weights_path = 'C:/Projects/Sparse_View/weights/' + train_name + '/'
    logs_path = 'C:/Projects/Sparse_View/logs/' + train_name + '/'
    # if not os.path.exists(save_weights_path):
    #     os.makedirs(save_weights_path)
    # if not os.path.exists(logs_path):
    #     os.makedirs(logs_path)

    #########################################################################
    # # load the data:
    with tf.name_scope('inputs'):
        pass
    training_iter_init, validation_iter_init, input_tensor, label_tensor = data_generator(
        file_train=FILE_TRAIN,
        file_valid=FILE_VALID,
        batch_size=BATCH_SIZE, buffer_size=2, x_size=SIZE, y_size=SIZE)
    #########################################################################


    #########################################################################
    # #  network
    model = Network().model
    pred_tensor = model(x_input=input_tensor, name=NET_NAME)
    #########################################################################


    #########################################################################
    # # loss and optimizer
    loss_function = Loss(BATCH_SIZE).loss
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LR, global_step,
                                                   LR_DECAY_STEP, LR_DECAY_RATE, staircase=True)
        loss = loss_function(y_pred=pred_tensor, y_truth=label_tensor)
        if OPTIMIZER == 'ADAM':
            train = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
        elif OPTIMIZER == 'SGD':
            train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        else:
            train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #########################################################################


    #########################################################################
    # # Evaluation Metrics
    with tf.name_scope('eval'):
        RMSE_eval_before, up_op_rmse_bef = Evaluation(BATCH_SIZE).RMSE(y_pred=input_tensor, y_truth=label_tensor)
        RMSE_eval_after, up_op_rmse_aft = Evaluation(BATCH_SIZE).RMSE(y_pred=pred_tensor, y_truth=label_tensor)
        SSIM_eval_before = Evaluation(BATCH_SIZE).SSIM(y_pred=input_tensor, y_truth=label_tensor)
        SSIM_eval_after = Evaluation(BATCH_SIZE).SSIM(y_pred=pred_tensor, y_truth=label_tensor)
    #########################################################################


    #####################################################################
    # Tensorboard part in the network below:
    # tf.summary.scalar("loss", loss)
    tf.summary.scalar("RMSE", RMSE_eval_after)
    tf.summary.scalar("LR", learning_rate)
    merged = tf.summary.merge_all()
    #####################################################################

    saver = tf.train.Saver(max_to_keep=5)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(training_iter_init)

        train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
        valid_writer = tf.summary.FileWriter(logs_path + '/valid', graph=tf.get_default_graph())

        if FLAG.RESTORE_GRAPH:
            saver.restore(sess, RESTORED_WEIGHT)
            print('Restoring Weights from:   ' + RESTORED_WEIGHT)
        else:
            print('Training from the scratch')
        print('Training Starts Now! ...............')
        for iepoch in range(EPOCH):
            #############################################
            # # each epoch Training:
            #############################################
            loss_tot, ssim_tot_before,  ssim_tot_after = 0, 0, 0
            for i_batch in range(int(np.ceil(TRAIN_SIZE / BATCH_SIZE))):

                _, _, _, loss_cal, ssim_cal_before, ssim_cal_after, summary, lr = \
                    sess.run([train, up_op_rmse_bef, up_op_rmse_aft, loss,  SSIM_eval_before, SSIM_eval_after, merged, learning_rate])
                loss_tot += loss_cal
                ssim_tot_before += ssim_cal_before
                ssim_tot_after += ssim_cal_after
                # #############################################
                # # # print out in-epoch progress:
                # if i_batch % int(np.ceil(TRAIN_SIZE / BATCH_SIZE)  / 8) == 0:
                #     localtime = time.asctime(time.localtime(time.time()))
                #     print(localtime, ':', "Epoch {} - {}/8...".format(iepoch+1, i_batch // int(np.ceil(TRAIN_SIZE / BATCH_SIZE)  / 8)),
                #           "General Loss: {:.7f} ...)".format(loss_tot / (i_batch+1)))
                # #############################################

            summary_value = tf.Summary()
            summary_value.value.add(tag='training/loss', simple_value=loss_tot/np.ceil(TRAIN_SIZE / BATCH_SIZE))
            summary_value.value.add(tag='training/ssim_after', simple_value=ssim_tot_after / np.ceil(TRAIN_SIZE / BATCH_SIZE))
            summary_value.value.add(tag='training/ssim_before', simple_value=ssim_tot_before / np.ceil(TRAIN_SIZE / BATCH_SIZE))
            summary_value.value.add(tag='training/rmse_before',simple_value= RMSE_eval_before.eval())
            train_writer.add_summary(summary_value, iepoch)
            train_writer.add_summary(summary, iepoch)

            #############################################
            # # print out progress:
            #############################################
            localtime = time.asctime(time.localtime(time.time()))
            PROGRESS = localtime + ':    ' + \
                       "Epoch {0:03}...".format(iepoch+1) + \
                       "General Loss: {:.7f} + RMSE: {:.7f} + LR: {:.7f}...".\
                           format(loss_tot / np.ceil(TRAIN_SIZE / BATCH_SIZE), RMSE_eval_after.eval(), lr)
            print(PROGRESS)
            # # output the progress to txt file:
            file_log = open(logs_path + 'log.txt', 'a+')
            file_log.write(PROGRESS + '\n')
            file_log.close()

            #############################################
            # Save out the model below
            #############################################
            if FLAG.SAVEWEIGHT and (iepoch+1) % 5 == 0:
                saver.save(sess, save_weights_path + 'unet3D.ckpt', global_step=iepoch+1)

            #############################################
            # # run the validation:
            #############################################
            if iepoch % 50 == 0:
                sess.run(validation_iter_init)
                loss_val_tot,  ssim_val_tot_before,  ssim_val_tot_after = 0, 0, 0
                for _ in range(int(np.ceil(VALID_SIZE / BATCH_SIZE))):
                    loss_val, _, _, ssim_val_before, ssim_val_after, summary = \
                        sess.run([loss, up_op_rmse_aft, up_op_rmse_bef ,SSIM_eval_before, SSIM_eval_after, merged])
                    loss_val_tot += loss_val
                    ssim_val_tot_before += ssim_val_before
                    ssim_val_tot_after += ssim_val_after

                summary_value = tf.Summary()
                summary_value.value.add(tag='validation/loss', simple_value=loss_val_tot / np.ceil(VALID_SIZE / BATCH_SIZE))
                summary_value.value.add(tag='validation/ssim_before', simple_value=ssim_val_tot_before / np.ceil(VALID_SIZE / BATCH_SIZE))
                summary_value.value.add(tag='validation/ssim_after', simple_value=ssim_val_tot_after / np.ceil(VALID_SIZE / BATCH_SIZE))
                summary_value.value.add(tag='validation/rmse_before', simple_value= RMSE_eval_before.eval())
                valid_writer.add_summary(summary_value, iepoch)
                valid_writer.add_summary(summary, iepoch)

                print('Validation Dataset: ', "Epoch {}...".format(iepoch),
                      "Loss: {:.7f},   RMSE: {:.7F}".format(loss_val_tot/np.ceil(VALID_SIZE / BATCH_SIZE), RMSE_eval_after.eval()))
                sess.run(training_iter_init)
                #######################################################
                # save the best model based on the validation evaluator:
                if iepoch == 0:
                    rmse_best = float('inf')
                if FLAG.SAVEWEIGHT and RMSE_eval_before.eval() < rmse_best:
                    rmse_best = RMSE_eval_before.eval()
                    saver.save(sess, save_weights_path + "best_rmse.ckpt" , global_step=iepoch+1)

            valid_writer.flush()
            train_writer.flush()
