import numpy as np
import tensorflow as tf
from basic_ops_p import *
import matplotlib.pyplot as plt
import os

# class TF_Records():
#
#     def __init__(self, file_path, ):

def data_generator(file_train = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset300_train195.tfrecords',
                   file_valid = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset300_valid30.tfrecords',
                   batch_size=1, buffer_size=8, x_size=1500*1024, y_size=1500*1024):

    def _parser_function(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # x_size = 512 * 360
        # y_size = 512 * 360

        keys_to_features = {
            'X': tf.FixedLenFeature([], tf.string),
            'Y': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(serialized_example, keys_to_features)

        x = tf.decode_raw(parsed_features['X'], tf.float32)
        y = tf.decode_raw(parsed_features['Y'], tf.float32)

        x_shape = tf.stack([x_size])
        y_shape = tf.stack([y_size])

        x = tf.reshape(x, x_shape)
        y = tf.reshape(y, y_shape)

        return x, y

    filenames_train = [file_train]
    dataset_train = tf.data.TFRecordDataset(filenames_train)
    dataset_train = dataset_train.map(_parser_function)
    dataset_train = dataset_train.shuffle(buffer_size=buffer_size)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.repeat()

    filenames_valid = [file_valid]
    dataset_valid = tf.data.TFRecordDataset(filenames_valid)
    dataset_valid = dataset_valid.map(_parser_function)
    dataset_valid = dataset_valid.shuffle(buffer_size=buffer_size)
    dataset_valid = dataset_valid.batch(batch_size=batch_size)
    dataset_valid = dataset_valid.repeat()

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    input_tensor, label_tensor = iterator.get_next()

    # Initialize with required Datasets
    iter_init_train = iterator.make_initializer(dataset_train)
    iter_init_valid = iterator.make_initializer(dataset_valid)

    return iter_init_train, iter_init_valid, input_tensor, label_tensor

def data_generator_test(file_name = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset300_test105.tfrecords',
                   batch_size=1, buffer_size=8, x_size=1500*1024, y_size=1500*1024):

    def _parser_function(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # x_size = 512 * 360 * 8
        # y_size = 512 * 360 * 8

        keys_to_features = {
            'X': tf.FixedLenFeature([], tf.string),
            'Y': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(serialized_example, keys_to_features)

        x = tf.decode_raw(parsed_features['X'], tf.float32)
        y = tf.decode_raw(parsed_features['Y'], tf.float32)

        x_shape = tf.stack([x_size])
        y_shape = tf.stack([y_size])

        x = tf.reshape(x, x_shape)
        y = tf.reshape(y, y_shape)

        return x, y

    filenames = [file_name]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parser_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()


    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    input_tensor, label_tensor = iterator.get_next()

    # Initialize with required Datasets
    iter_init = iterator.make_initializer(dataset)

    return iter_init, input_tensor, label_tensor

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_one_record(file = 'test_2example.tfrecords', x_size=4*1024, y_size=4*1024):

    filename_queue = tf.train.string_input_producer([file], num_epochs=10)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'X': tf.FixedLenFeature([], tf.string),
            'Y': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    x = tf.decode_raw(features['X'], tf.float32)
    y = tf.decode_raw(features['Y'], tf.float32)

    x_shape = tf.stack([1, x_size])
    y_shape = tf.stack([1, y_size])

    x = tf.reshape(x, x_shape)
    y = tf.reshape(y, y_shape)

    return x, y

def np_to_tfrecords(X, Y, file_name):
    writer = tf.python_io.TFRecordWriter(file_name)

    print("Data Writing Starts...... Total number of images to write is: ", str(X.shape[0]), '.............')
    for idx in range(X.shape[0]):
        if idx % 50 == 0:
            print(idx)

        x = X[idx]
        y = Y[idx]
        x_raw = x.tostring()
        y_raw = y.tostring()

        keys_to_features = {'X': _bytes_feature(x_raw),
                            'Y': _bytes_feature(y_raw)}
        example = tf.train.Example(features=tf.train.Features(feature= keys_to_features))
        writer.write(example.SerializeToString())

    print("Data Writing Finished...... Total number of written images is: ", str(idx + 1), '.............')
    writer.close()

def readMatData_sino(file_paths = ['C:/Projects/PCCT_Recon/data/simulation data5/sino/'],
                start_slices=[200],  end_slices=[800], sample_rate =[5], divided=[70, 30], row = 1500, column = 1024):
    # read the mat data using the original code below:
    import scipy.io
    size = row * column
    num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))])
    feature_img = np.zeros((num_img, size), dtype=np.float32)
    label_img = np.zeros((num_img, size), dtype=np.float32)

    i_img = 0
    for i_scan in range(len(start_slices)):
        for i_slice in range(start_slices[i_scan], end_slices[i_scan], sample_rate[i_scan]):
            feature_img[i_img, :] = transform_to_2D_np(scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_inter.mat')[
                    'img_sino_inter']).astype(np.float32)
            label_img[i_img, :] = transform_to_2D_np(scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_truth.mat')[
                'img_sino_truth']).astype(np.float32)
            i_img += 1
    feature_img[np.isnan(feature_img)] = 0
    label_img[np.isnan(label_img)] = 0
    arr_shuf = np.arange(num_img)
    np.random.shuffle(arr_shuf)
    feature_img = feature_img[arr_shuf,:]
    label_img = label_img[arr_shuf,:]
    if len(divided) <= 1:
        return feature_img, label_img
    if len(divided) == 2:
        num_img_1 = int(num_img * divided[0] / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], feature_img[num_img_1::, :], label_img[num_img_1::, :]
    if len(divided) == 3:
        num_img_1 = int(num_img * divided[0] / 100.0)
        num_img_2 = int(num_img * (divided[0] + divided[1]) / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], \
               feature_img[num_img_1:num_img_2, :], label_img[num_img_1:num_img_2,:], \
               feature_img[num_img_2::, :], label_img[num_img_2::, :]


def readMatData_patch(file_paths = ['C:/Projects/PCCT_Recon/data/simulation data5/sino/'],
                start_slices=[200],  end_slices=[800], sample_rate =[5], divided=[70, 30], patch_row = 60, patch_col = 64, row = 1500, column = 1024):
    # read the mat data using the original code below:
    import scipy.io
    size = patch_row * patch_col
    num_patch_row = int(np.ceil(row/patch_row))
    num_patch_col = int(np.ceil(column/patch_col))

    num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))]) * num_patch_row * num_patch_col
    feature_img = np.zeros((num_img, size), dtype=np.float32)
    label_img = np.zeros((num_img, size), dtype=np.float32)

    i_img = 0
    for i_scan in range(len(start_slices)):
        for i_slice in range(start_slices[i_scan], end_slices[i_scan], sample_rate[i_scan]):
            feature= scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_inter.mat')[
                    'img_sino_inter'].astype(np.float32)
            label = scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_truth.mat')[
                'img_sino_truth'].astype(np.float32)
            for i_row in range(num_patch_row):
                for i_col in range(num_patch_col):
                    if (i_row+1)*patch_row > row and (i_col+1)*patch_col>column:
                        feature_img[i_img, :] = transform_to_2D_np(feature[row - patch_row: row, column - patch_col: column])
                        label_img[i_img, :] = transform_to_2D_np( label[row - patch_row: row, column - patch_col: column])
                    elif (i_row+1)*patch_row > row:
                        feature_img[i_img, :] = transform_to_2D_np(feature[row - patch_row: row,
                                                                   i_col * patch_col: (i_col + 1) * patch_col])
                        label_img[i_img, :] = transform_to_2D_np(label[row - patch_row: row,
                                                                 i_col * patch_col: (i_col + 1) * patch_col])
                    elif (i_col+1)*patch_col>column:
                        feature_img[i_img, :] = transform_to_2D_np(feature[i_row * patch_row: (i_row + 1) * patch_row,
                                                                   column - patch_col: column])
                        label_img[i_img, :] = transform_to_2D_np(label[i_row * patch_row: (i_row + 1) * patch_row,
                                                                 column - patch_col: column])
                    else:
                        feature_img[i_img, :] = transform_to_2D_np(feature[i_row*patch_row : (i_row+1)*patch_row,  i_col*patch_col : (i_col+1)*patch_col])
                        label_img[i_img, :] = transform_to_2D_np(label[i_row*patch_row : (i_row+1)*patch_row,  i_col*patch_col : (i_col+1)*patch_col])
                    i_img += 1
    feature_img[np.isnan(feature_img)] = 0
    label_img[np.isnan(label_img)] = 0
    arr_shuf = np.arange(num_img)
    np.random.shuffle(arr_shuf)
    feature_img = feature_img[arr_shuf,:]
    label_img = label_img[arr_shuf,:]
    if len(divided) <= 1:
        return feature_img, label_img
    if len(divided) == 2:
        num_img_1 = int(num_img * divided[0] / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], feature_img[num_img_1::, :], label_img[num_img_1::, :]
    if len(divided) == 3:
        num_img_1 = int(num_img * divided[0] / 100.0)
        num_img_2 = int(num_img * (divided[0] + divided[1]) / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], \
               feature_img[num_img_1:num_img_2, :], label_img[num_img_1:num_img_2,:], \
               feature_img[num_img_2::, :], label_img[num_img_2::, :]

def readMatData_patch2(file_paths = ['C:/Projects/PCCT_Recon/data/simulation data5/sino/'],
                start_slices=[200],  end_slices=[800], sample_rate =[5], divided=[70, 30],
                       patch_row = 64, patch_col = 64, row = 1500, column = 1024, num_patch_row=50, num_patch_col=50):

    # read the mat data using the original code below:
    import scipy.io
    size = patch_row * patch_col
    row_index = np.linspace(start=0, stop=row-patch_row, num=num_patch_row, endpoint=True).astype('int')
    col_index = np.linspace(start=0, stop=column - patch_col, num=num_patch_col, endpoint=True).astype('int')
    num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))]) * num_patch_row * num_patch_col
    feature_img = np.zeros((num_img, size), dtype=np.float32)
    label_img = np.zeros((num_img, size), dtype=np.float32)

    i_img = 0
    for i_scan in range(len(start_slices)):
        for i_slice in range(start_slices[i_scan], end_slices[i_scan], sample_rate[i_scan]):
            feature= scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_inter.mat')[
                    'img_sino_inter'].astype(np.float32)
            label = scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_truth.mat')[
                'img_sino_truth'].astype(np.float32)
            for i_row in row_index:
                for i_col in col_index:
                    feature_img[i_img, :] = transform_to_2D_np(feature[i_row : i_row + patch_row,  i_col : i_col + patch_col])
                    label_img[i_img, :] = transform_to_2D_np(label[i_row : i_row + patch_row,  i_col : i_col + patch_col])
                    i_img += 1
    feature_img[np.isnan(feature_img)] = 0
    label_img[np.isnan(label_img)] = 0
    arr_shuf = np.arange(num_img)
    np.random.shuffle(arr_shuf)
    feature_img = feature_img[arr_shuf,:]
    label_img = label_img[arr_shuf,:]
    if len(divided) <= 1:
        return feature_img, label_img
    if len(divided) == 2:
        num_img_1 = int(num_img * divided[0] / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], feature_img[num_img_1::, :], label_img[num_img_1::, :]
    if len(divided) == 3:
        num_img_1 = int(num_img * divided[0] / 100.0)
        num_img_2 = int(num_img * (divided[0] + divided[1]) / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], \
               feature_img[num_img_1:num_img_2, :], label_img[num_img_1:num_img_2,:], \
               feature_img[num_img_2::, :], label_img[num_img_2::, :]

def readMatData_patch3(file_paths = ['C:/Projects/PCCT_Recon/data/simulation data5/sino/'],
                start_slices=[200],  end_slices=[800], sample_rate =[5], divided=[70, 30],
                       row=[1500], column=[1024], num_patch_row=[50], num_patch_col=[50], patch_row = 64, patch_col = 64):

    # read the mat data using the original code below:
    import scipy.io
    size = patch_row * patch_col
    num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i]  for i in range(len(start_slices))])
    feature_img = np.zeros((num_img, size), dtype=np.float32)
    label_img = np.zeros((num_img, size), dtype=np.float32)

    i_img = 0
    for i_scan in range(len(start_slices)):
        print('i_scan:', i_scan)
        for i_slice in range(start_slices[i_scan], end_slices[i_scan], sample_rate[i_scan]):
            feature= scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_inter.mat')[
                    'img_sino_inter'].astype(np.float32)
            label = scipy.io.loadmat(file_paths[i_scan] + 'slice_' + str(i_slice).zfill(4) + '_truth.mat')[
                'img_sino_truth'].astype(np.float32)

            # row_index = np.random.choice(row[i_scan] - patch_row + 1, num_patch_row[i_scan], replace=False).astype('int')
            # col_index = np.random.choice(column[i_scan] - patch_col + 1, num_patch_col[i_scan], replace=False).astype('int')
            row_index = np.linspace(start=0, stop=row[i_scan] - patch_row, num=num_patch_row[i_scan], endpoint=True).astype('int')
            col_index = np.linspace(start=0, stop=column[i_scan] - patch_col, num=num_patch_col[i_scan], endpoint=True).astype('int')
            for i_row in row_index:
                for i_col in col_index:
                    feature_img[i_img, :] = transform_to_2D_np(feature[i_row : i_row + patch_row,  i_col : i_col + patch_col])
                    label_img[i_img, :] = transform_to_2D_np(label[i_row : i_row + patch_row,  i_col : i_col + patch_col])
                    i_img += 1
    feature_img[np.isnan(feature_img)] = 0
    label_img[np.isnan(label_img)] = 0
    arr_shuf = np.arange(num_img)
    np.random.shuffle(arr_shuf)
    feature_img = feature_img[arr_shuf,:]
    label_img = label_img[arr_shuf,:]
    if len(divided) <= 1:
        return feature_img, label_img
    if len(divided) == 2:
        num_img_1 = int(num_img * divided[0] / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], feature_img[num_img_1::, :], label_img[num_img_1::, :]
    if len(divided) == 3:
        num_img_1 = int(num_img * divided[0] / 100.0)
        num_img_2 = int(num_img * (divided[0] + divided[1]) / 100.0)
        return feature_img[0:num_img_1, :], label_img[0:num_img_1, :], \
               feature_img[num_img_1:num_img_2, :], label_img[num_img_1:num_img_2,:], \
               feature_img[num_img_2::, :], label_img[num_img_2::, :]


def testNumpyData(feature_img, label_img, row = 1500, column = 1024):
    feature_img = np.reshape(feature_img, [row, column])
    label_img = np.reshape(label_img, [row, column])
    contrast_display = (np.min(label_img), np.max(label_img))
    plt.figure()
    plt.subplot(121)
    plt.imshow(feature_img, vmin=contrast_display[0], vmax=contrast_display[1])
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(label_img, vmin=contrast_display[0], vmax=contrast_display[1])
    plt.colorbar()
    plt.show()


# def sliptData_random(range, num_images, num_train, num_valid, num_test,
#                      cvs_file_name = 'C:/Projects/PCCT_data/simulation data/TFRecord_img/data_split_256.csv',
#                      file_path='C:/Projects/PCCT_data/data/simulation data2/sino/',
#                      row=1024, total_views = 720, sample_view =2,  bin = 4):
#     '''
#     This function is to randomly choose num_images images and randomly split into train, valiation, and test
#
#     :param num_images:
#     :param num_train:
#     :param num_valid:
#     :param num_test:
#     :param feature_img:
#     :param label_img:
#     :return:
#     '''
#     ind = np.arange(range, dtype='int32')
#     np.random.shuffle(ind)
#     ind = ind[0:num_images]
#     train_index = ind[0 : num_train]
#     valid_index = ind[num_train : num_valid + num_train]
#     test_index = ind[num_valid + num_train : num_test + num_valid + num_train]
#
#     feature_img_train, label_img_train = readMatData_sino3(file_path = file_path, image_list = train_index,
#                                                            row=row, total_views=total_views, sample_view=sample_view, bin=bin)
#     feature_img_valid, label_img_valid = readMatData_sino3(file_path=file_path, image_list=valid_index,
#                                                            row=row, total_views=total_views, sample_view=sample_view, bin=bin)
#     feature_img_test, label_img_test = readMatData_sino3(file_path=file_path, image_list=test_index,
#                                                          row=row, total_views=total_views, sample_view=sample_view, bin=bin)
#
#     # save out the index:
#     import pandas as pd
#     df1 = pd.DataFrame({'Train': train_index})
#     df2 = pd.DataFrame({'Valid': valid_index})
#     df3 = pd.DataFrame({'Test': test_index})
#     df = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
#     print(df)
#     df.to_csv(cvs_file_name, header=['Train', 'Valid', 'Test'] ,index=True)
#
#     return feature_img_train, label_img_train, feature_img_valid, label_img_valid, feature_img_test, label_img_test
#
#
# def sliptData_random_fulltest(range, num_images, num_train, num_valid, num_test,
#                      cvs_file_name = 'C:/Projects/PCCT_data/simulation data/TFRecord_img/data_split_256.csv',
#                      file_path='C:/Projects/PCCT_data/data/simulation data2/sino/',
#                      row=1024, total_views = 720, sample_view =2,  bin = 4):
#     '''
#     This function is to randomly choose num_images images and randomly split into train, valiation, and test
#
#     for the test, make the sample_view always equal to 1 (To get the full images.)
#
#     :param num_images:
#     :param num_train:
#     :param num_valid:
#     :param num_test:
#     :param feature_img:
#     :param label_img:
#     :return:
#     '''
#     ind = np.arange(range, dtype='int32')
#     np.random.shuffle(ind)
#     ind = ind[0:num_images]
#     train_index = ind[0 : num_train]
#     valid_index = ind[num_train : num_valid + num_train]
#     test_index = ind[num_valid + num_train : num_test + num_valid + num_train]
#
#     feature_img_train, label_img_train = readMatData_sino3(file_path = file_path, image_list = train_index,
#                                                            row=row, total_views=total_views, sample_view=sample_view, bin=bin)
#     feature_img_valid, label_img_valid = readMatData_sino3(file_path=file_path, image_list=valid_index,
#                                                            row=row, total_views=total_views, sample_view=sample_view, bin=bin)
#     feature_img_test, label_img_test = readMatData_sino3(file_path=file_path, image_list=test_index,
#                                                          row=row, total_views=total_views, sample_view=1, bin=bin)
#
#     # save out the index:
#     import pandas as pd
#     # df0 = pd.DataFrame({'Num': np.arange(max(train_index.size, valid_index.size, test_index.size), dtype='int32')})
#     df1 = pd.DataFrame({'Train': train_index})
#     df2 = pd.DataFrame({'Valid': valid_index})
#     df3 = pd.DataFrame({'Test': test_index})
#     df = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
#     print(df)
#     df.to_csv(cvs_file_name, header=['Train', 'Valid', 'Test'] ,index=True)
#
#     return feature_img_train, label_img_train, feature_img_valid, label_img_valid, feature_img_test, label_img_test



# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ## Testing Functions Below:
# def test_tfrecord_data(tf_records_file = 'C:/Projects/PCCT_Recon/data/simulation data2/TFRecord_sino/test_2example.tfrecords'):
#     iter_init_train, iter_init_valid, x, y = data_generator(file_train=tf_records_file, file_valid=tf_records_file)
#
#     x_tran = transform_to_4D_tf(x)
#     x_tran = transform_to_2D_tf(x_tran)
#     x_tran = transform_to_4D_tf(x_tran)
#
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         sess.run(iter_init_train)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         for _ in range(4):  # run a few epoch here:
#             x_out, y_out = sess.run([x, y])
#             print(x_out.dtype, y_out.dtype)
#
#         x_out = y_out # check y data
#         x_img = transform_to_2D_np(x_out[0, ...])
#         plt.figure()
#         plt.imshow(x_img, cmap='gray', vmin=0, vmax=np.max(x_img))
#         plt.colorbar()
#         plt.axes()
#         plt.show()
#
#         x_tran_out = sess.run(x_tran)
#         print(x_tran_out.shape)
#         plt.figure()
#         plt.imshow(np.squeeze(x_tran_out[0, ..., 0]), cmap='gray', vmin=0, vmax=np.max(x_tran_out[0, ..., 0]))
#         plt.colorbar()
#         plt.show()
#
#         return
#
# def compare_tfrecord_matdata(tf_records_file = 'C:/Projects/PCCT_Recon/data/simulation data2/TFRecord_sino/test_2example.tfrecords',
#                              file_path = 'C:/Projects/PCCT_Recon/data/simulation data2/sino/',
#                              row=512, column = 360, bin = 8):
#     iter_init_train, iter_init_valid, x, y = data_generator(file_train=tf_records_file, file_valid=tf_records_file)
#
#
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         sess.run(iter_init_train)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         x_out, y_out = sess.run([x, y])
#     img_py = transform_to_3D_np(x_out)
#     import os
#     import scipy.io
#     size = row * column * bin
#     i_name = 0
#     img_mat = np.transpose(
#             scipy.io.loadmat(file_path + str(i_name).zfill(4) + '_sino_log_real_mean.mat')['proj_attlog_real_mean'][:, 0:512,:], (1, 2, 0))
#     # print(x_out.shape)
#     # print(img_mat)
#
#
#
#     ibin = 0
#     # plt.figure()
#     # plt.imshow(np.squeeze(img_py[:, :, ibin]), cmap='gray', vmin=0, vmax=np.max(img_py[:, :, ibin]))
#     # plt.show()
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.squeeze(img_py[:,:,ibin]), cmap='gray', vmin=0, vmax=np.max(img_py[:,:,ibin]))
#     plt.colorbar()
#     plt.subplot(122)
#     plt.imshow(np.squeeze(img_mat[:, :, ibin]), cmap='gray', vmin=0, vmax=np.max(img_py[:, :, ibin]))
#     plt.colorbar()
#
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.squeeze(img_py[:,:,ibin] - img_mat[:, :, ibin]), cmap='jet', vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.subplot(122)
#     plt.imshow(np.squeeze((img_py[:, :, ibin] - img_mat[:, :, ibin])/img_mat[:, :, ibin]), cmap='jet', vmin=-0.01, vmax=0.01)
#     plt.colorbar()
#     plt.show()
#
#
# def compare_tfrecord_matdata2(file_path = 'C:/Projects/PCCT_data/simulation data5/sino/',
#                              row=512, column = 360, bin = 8):
#     '''
#     This is for the view data comparison.
#     '''
#
#     # read the mat data:
#     import os
#     import scipy.io
#     size = row * column * bin
#     i_name = 0
#     x_out_mat, y_out_mat = readMatData_sino3(file_path=file_path,
#                                              image_list=np.arange(1), row=512, total_views=360, sample_view=1, bin=4)
#     img_mat_x = transform_to_3D_np(x_out_mat)
#     img_mat_y = transform_to_3D_np(y_out_mat)
#
#     # save the mat data to tfrecords:
#     tf_records_file = 'C:/Projects/PCCT_data/simulation data5/TFRecord_view/compare_tfrecord_mat.tfrecords'
#     np_to_tfrecords(x_out_mat, y_out_mat, tf_records_file)
#
#
#     iter_init_train, iter_init_valid, x, y = data_generator(file_train=tf_records_file, file_valid=tf_records_file, batch_size=360, buffer_size=1)
#
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         sess.run(iter_init_train)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         x_out, y_out = sess.run([x, y])
#     img_py_x = transform_to_3D_np(x_out)
#     img_py_y = transform_to_3D_np(y_out)
#
#     ibin = 0
#     # plt.figure()
#     # plt.imshow(np.squeeze(img_py[:, :, ibin]), cmap='gray', vmin=0, vmax=np.max(img_py[:, :, ibin]))
#     # plt.show()
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.squeeze(img_py_x[:,ibin,:]), cmap='gray', vmin=0, vmax=np.max(img_mat_x[:,ibin,:]))
#     plt.colorbar()
#     plt.subplot(122)
#     plt.imshow(np.squeeze(img_mat_x[:,ibin,:]), cmap='gray', vmin=0, vmax=np.max(img_mat_x[:,ibin,:]))
#     plt.colorbar()
#
#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(np.squeeze(img_py_x[:,ibin,:] - img_mat_x[:,ibin,:] ), cmap='jet', vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.subplot(122)
#     plt.imshow(np.squeeze((img_py_x[:,ibin,:] - img_mat_x[:,ibin,:] )/img_mat_x[:,ibin,:] ), cmap='jet', vmin=-0.01, vmax=0.01)
#     plt.colorbar()
#     plt.show()




## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## do the test
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
if __name__ == "__main__":
    pass

    # ###################################################################################################
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/'
    # ]
    # start_slices = [512-10, 512-10]
    # end_slices =   [512+30, 512+30]
    # sample_rate = [20, 20]
    # divided = [100]
    # row = 1500
    # column = 1024
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = 5
    # num_patch_col = 5
    # num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))])* num_patch_row * num_patch_col
    #
    # feature_img_train, label_img_train = \
    #     readMatData_patch2(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices,
    #                        sample_rate=sample_rate, divided=divided,
    #                        patch_row=patch_row, patch_col=patch_col, row=row, column=column,
    #                        num_patch_row=num_patch_row, num_patch_col=num_patch_col)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p_test'\
    #                   + str(int(num_img)) + '_size' + str(patch_row) + 'x' + str(patch_col) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)

    ####################################################################################################
    # # # This is for the training and validation
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/'
    # ]
    # start_slices = [512-150, 512-150]
    # end_slices = [512+150, 512+150]
    # sample_rate = [6, 6]
    # divided = [65, 10, 35]
    # row = 1500
    # column = 1024
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = 25
    # num_patch_col = 25
    # num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))])* num_patch_row * num_patch_col
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_train, label_img_train, feature_img_valid, label_img_valid, feature_img_test, label_img_test = \
    #     readMatData_patch2(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate, divided=divided,
    #                        patch_row = patch_row, patch_col = patch_col, row=row, column=column, num_patch_row=num_patch_row, num_patch_col=num_patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_train' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_valid' + str(int(num_img*divided[1]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_valid, label_img_valid, tf_records_file)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p' \
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_test' + str(int(num_img*divided[2]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_test, label_img_test, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_train[i_patch,], label_img_train[i_patch,], row=patch_row, column=patch_col)


    # ####################################################################################################
    # # # This is for the training dataset using readMatData_patch3
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_3/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_4/'
    # ]
    # start_slices = [800, 300, 300, 200]
    # end_slices = [1248, 724, 724, 1826]
    # sample_rate = [10, 4, 4, 24]
    # divided = [100]
    # row = [550, 1500, 364, 600]
    # column = [2048, 1024, 1024, 2026]
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = [15, 20, 15, 15]
    # num_patch_col = [20, 15, 15, 20]
    # num_img = sum(
    #     [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
    #      range(len(start_slices))])
    #
    # feature_img_train, label_img_train = \
    #     readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_train' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_train[i_patch,], label_img_train[i_patch,], row=patch_row, column=patch_col)

    # ####################################################################################################
    # # # This is for the validation dataset
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_7/',
    # ]
    # start_slices = [300]
    # end_slices = [724]
    # sample_rate = [16]
    # divided = [100]
    # row = [689]
    # column = [1024]
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = [12]
    # num_patch_col = [25]
    # num_img = sum(
    #     [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
    #      range(len(start_slices))])
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_valid, label_img_valid = \
    #     readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_valid' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_valid, label_img_valid, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_valid[i_patch,], label_img_valid[i_patch,], row=patch_row, column=patch_col)


    ####################################################################################################
    # # # This is for the training and validation dataset using readMatData_patch3
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_3/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_4/'
    # ]
    # start_slices = [800, 300, 350, 200]
    # end_slices = [1248, 724, 674, 1826]
    # sample_rate = [10, 4, 4, 24]
    # divided = [90, 10]
    # row = [550, 1500, 364, 600]
    # column = [2048, 955, 1024, 2026]
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = [15, 20, 15, 15]
    # num_patch_col = [20, 15, 15, 20]
    # num_img = sum(
    #     [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
    #      range(len(start_slices))])
    #
    # feature_img_train, label_img_train, feature_img_valid, label_img_valid  = \
    #     readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_train' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_valid' + str(int(num_img*divided[1]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_valid, label_img_valid, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_train[i_patch,], label_img_train[i_patch,], row=patch_row, column=patch_col)

    # ####################################################################################################
    # # # This is for the test dataset
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_7/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_8/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_9/'
    # ]
    # start_slices = [300, 600, 600]
    # end_slices = [724, 1448, 1448]
    # sample_rate = [3, 12, 9]
    # divided = [100]
    # row = [689, 360, 500]
    # column = [825, 1849, 1609]
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = [12, 6, 8]
    # num_patch_col = [18, 35, 35]
    # num_img = sum(
    #     [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
    #      range(len(start_slices))])
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_test, label_img_test = \
    #     readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_test' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_test, label_img_test, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_test[i_patch,], label_img_test[i_patch,], row=patch_row, column=patch_col)

    # ####################################################################################################
    # # # This is only one scan dataset for testing program
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_7/'
    # ]
    # start_slices = [380]
    # end_slices = [640]
    # sample_rate = [1]
    # divided = [100]
    # row = [689]
    # column = [825]
    # patch_row = 64
    # patch_col = 64
    # num_patch_row = [10]
    # num_patch_col = [10]
    # num_img = sum(
    #     [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
    #      range(len(start_slices))])
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_test, label_img_test = \
    #     readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
    #                   + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
    #                   + '_test' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_test, label_img_test, tf_records_file)
    #
    # i_patch = 1000
    # testNumpyData(feature_img_test[i_patch,], label_img_test[i_patch,], row=patch_row, column=patch_col)

    ####################################################################################################
    # # This is only one slice image in one scan for testing purpose
    import time
    start_time = time.time()

    file_paths = [
                    'C:/Projects/Sparse_View_data/sinogram_data/scan_7/'
    ]
    start_slices = [511]
    end_slices = [512]
    sample_rate = [1]
    divided = [100]
    row = [689]
    column = [825]
    patch_row = 64
    patch_col = 64
    num_patch_row = [20]
    num_patch_col = [20]
    num_img = sum(
        [len(range(start_slices[i], end_slices[i], sample_rate[i])) * num_patch_row[i] * num_patch_col[i] for i in
         range(len(start_slices))])

    # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    #                      divided=divided, row=row, column=column)
    feature_img_test, label_img_test = \
        readMatData_patch3(file_paths, start_slices,  end_slices, sample_rate, divided, row, column, num_patch_row, num_patch_col, patch_row, patch_col)
    print("--- %s seconds ---" % (time.time() - start_time))

    tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_p'\
                      + str(int(num_img))+ '_size' + str(patch_row) + 'x' + str(patch_col)  \
                      + '_test' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    np_to_tfrecords(feature_img_test, label_img_test, tf_records_file)

    i_patch = 10
    testNumpyData(feature_img_test[i_patch,], label_img_test[i_patch,], row=patch_row, column=patch_col)

    ####################################################################################################
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/'
    # ]
    # start_slices = [512-10, 512-10]
    # end_slices =   [512+30, 512+30]
    # sample_rate = [20, 20]
    # divided = [100]
    # row = 1500
    # column = 1024
    # num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))])
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_train, label_img_train = \
    #     readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    #                      divided=divided, row=row, column=column)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset_test'\
    #                   + str(int(num_img)) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)

    ####################################################################################################
    # import time
    # start_time = time.time()
    #
    # file_paths = [
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_1/',
    #                 'C:/Projects/Sparse_View_data/sinogram_data/scan_2/'
    # ]
    # start_slices = [512-150, 512-150]
    # end_slices = [512+150, 512+150]
    # sample_rate = [6, 6]
    # divided = [65, 10, 35]
    # row = 1500
    # column = 1024
    # num_img = sum([len(range(start_slices[i], end_slices[i], sample_rate[i])) for i in range(len(start_slices))])
    #
    # # feature_img_train, label_img_train = readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    # #                      divided=divided, row=row, column=column)
    # feature_img_train, label_img_train, feature_img_valid, label_img_valid, feature_img_test, label_img_test = \
    #     readMatData_sino(file_paths=file_paths, start_slices=start_slices, end_slices=end_slices, sample_rate=sample_rate,
    #                      divided=divided, row=row, column=column)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset'\
    #                   + str(int(num_img)) + '_train' + str(int(num_img*divided[0]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_train, label_img_train, tf_records_file)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset'\
    #                   + str(int(num_img)) + '_valid' + str(int(num_img*divided[1]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_valid, label_img_valid, tf_records_file)
    # tf_records_file = 'C:/Projects/Sparse_View_data/TFRecord_sino/dataset' \
    #                   + str(int(num_img)) + '_test' + str(int(num_img*divided[2]/100.)) + '.tfrecords'
    # np_to_tfrecords(feature_img_test, label_img_test, tf_records_file)

    #################################################################################
    # ## saving the tfrecords data from .mat images
    # file_path = 'C:/Projects/PCCT_data/simulation data5/sino/'
    # num_images = 1
    # total_views = 360
    # sample_view = 10
    # num_examples = int(num_images * total_views / sample_view)
    # x_data, y_data = readMatData_sino3(file_path = file_path,
    #                                    image_list=np.arange(num_images), row=512, total_views=total_views, sample_view=sample_view, bin=4)
    # tf_records_file = 'C:/Projects/PCCT_data/simulation data6/TFRecord_view/test_'+ str(num_examples) + 'example.tfrecords'
    # np_to_tfrecords(x_data, y_data, tf_records_file)

    ####################################################################################################
    ## Check the readMatData function:
    # Read saved tfrecords data, and show:
    # tf_records_file = 'C:/Projects/PCCT_data/simulation data18/TFRecord_view/dataset352_train256_views3072_proc.tfrecords'
    # test_tfrecord_data(tf_records_file=tf_records_file)
    #################################################################################


    #################################################################################
    # generate np random number for test:
    # x_data = np.random.random_sample((2, 512*512*8)).astype('float32')
    # y_data = np.random.random_sample(x_data.shape).astype('float32')
    # tf_records_file = 'C:/Projects/PCCT_Recon/data/simulation data/TFRecord_img/test_random_2example.tfrecords'
    # np_to_tfrecords(x_data, y_data, tf_records_file)

    #################################################################################
    # compare_tfrecord_matdata2()


