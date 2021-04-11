########################################################################################################################
####    This is the file containing tools we would use for training and testing                                     ####
########################################################################################################################

import parameters
import random
import numpy as np
import os
from scipy.misc import imread, imsave
import rawpy


class tools:
    def __init__(self):
        self.epoch = parameters.EPOCH_NUM
        self.real_data_path = parameters.REAL_DATA_PATH
        self.training_data_path = parameters.TRAINING_DATA_PATH
        self.testing_data_path = parameters.TESTING_DATA_PATH
        self.batch_size = parameters.BATCH_SIZE
        self.training_train_file = parameters.TRAINING_TRAIN_FILE
        self.subfolder_groundtruth = parameters.SUBFOLDER_GROUNDTRUTH
        self.subfolder_trainingdata = parameters.SUBFOLDER_TRAININGDATA
        self.subfolder_isp = parameters.SUBFOLDER_ISP
        self.crop_size = parameters.CROP_SIZE
        self.test_train_file = parameters.TESTING_TRAIN_FILE
        self.result_path = parameters.RESULT_PATH
        self.file_report = parameters.FILENAME_REPORT
        self.test_image_folder = parameters.TEST_IMAGE_FOLDER
        self.test_step = parameters.TEST_STEP
        self.train_subfolder = 'subfolder'
        self.folder_num = 100

    def gamma(self, image):
        image = image.astype(np.float32)/255.
        y = (1 + 0.055) * np.power(image, 1 / 2.4) - 0.055
        y[image < 0.0031308] = 12.92 * image[image < 0.0031308]
        return y

    def bayerprocess(self, rgb):
        rgb = rgb.astype(np.float32)/65535.
        bayer = np.copy(rgb[:, :, 0])
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer

    def new_shape(self, shape):
        if len(shape) == 2:
            return [shape[0]/2*2, shape[1]/2*2]
        else:
            return [shape[0], shape[1]/2*2, shape[2]/2*2, shape[3]]

    def load_test_data(self, batch_index):
        train_data = []
        isp_data = []
        name_train = self.test_train_file[batch_index]
        name_train = name_train[:-3]
        name = name_train[:-5]+'.'
        scale = parameters.Scale
        train_img = np.load(os.path.join(self.testing_data_path, self.subfolder_trainingdata, name_train + 'npy'))
        ground_img = np.load(os.path.join(self.testing_data_path, self.subfolder_groundtruth, name+'npy'))
        isp_img = imread(os.path.join(self.testing_data_path, self.subfolder_isp, name_train+'jpeg')).astype(np.float32)/255.
        h = train_img.shape[0] // 2 * 2
        w = train_img.shape[1] // 2 * 2
        train_img = np.expand_dims(train_img, axis=2)[:h, :w, :]
        ground_img = ground_img[:scale * h, :scale * w, :]
        isp_img = isp_img[:h, :w, :]
        mask = np.zeros(shape=[scale * h, scale * w, 3])
        for i in range(0, h - self.crop_size + self.test_step, self.test_step):
            for j in range(0, w - self.crop_size + self.test_step, self.test_step):
                i = min(h - self.crop_size, i)
                j = min(w - self.crop_size, j)
                ie = i + self.crop_size
                je = j + self.crop_size
                mask[scale * i:scale * ie, scale * j:scale * je, :] += 1
                t_data = train_img[i:ie, j:je, :]
                i_data = isp_img[i:ie, j:je, :]
                train_data.append(t_data)
                isp_data.append(i_data)
        return np.stack(train_data), np.stack(isp_data), h, w, name_train, ground_img, mask

    def load_test_real(self, batch_index):
        train_data = []
        isp_data = []
        name_train = self.test_train_file[batch_index]
        raw = rawpy.imread(os.path.join(self.real_data_path, name_train))
        rgb = raw.postprocess(gamma=(1, 1), user_wb=[1, 1, 1, 1], output_bps=16, output_color=rawpy.ColorSpace.raw,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, no_auto_scale=False, no_auto_bright=True)
        name_train = name_train.split('.')[0]
        train_img = self.bayerprocess(rgb=rgb)
        isp_ungamma = raw.postprocess(gamma=(1, 1))
        isp_img = self.gamma(isp_ungamma)
        h = train_img.shape[0] // 2 * 2
        w = train_img.shape[1] // 2 * 2
        train_img = np.expand_dims(train_img, axis=2)[:h, :w, :]
        isp_img = isp_img[:h, :w, :]
        mask = np.zeros(shape=[2 * h, 2 * w, 3])
        for i in range(0, h - self.crop_size + self.test_step, self.test_step):
            for j in range(0, w - self.crop_size + self.test_step, self.test_step):
                i = min(h - self.crop_size, i)
                j = min(w - self.crop_size, j)
                ie = i + self.crop_size
                je = j + self.crop_size
                mask[2 * i:2 * ie, 2 * j:2 * je, :] += 1
                t_data = train_img[i:ie, j:je, :]
                i_data = isp_img[i:ie, j:je, :]
                train_data.append(t_data)
                isp_data.append(i_data)
        self.imgsave(isp_img*255., self.epoch, '{}_lowresolution.'.format(name_train))
        return np.stack(train_data), np.stack(isp_data), h, w, name_train, mask

    def merge(self, res_list, h, w, mask):
        scale = parameters.Scale
        res = np.zeros([scale * h, scale * w, 3], dtype=np.float32)
        index = 0
        for i in range(0, scale * (h - self.crop_size + self.test_step), scale * self.test_step):
            for j in range(0, scale * (w - self.crop_size + self.test_step), scale * self.test_step):
                i = min(scale * (h - self.crop_size), i)
                j = min(scale * (w - self.crop_size), j)
                ie = i + scale * self.crop_size
                je = j + scale * self.crop_size
                res[i:ie, j:je, :] += res_list[index, :ie - i, :je - j, :]
                index += 1
        res = res / mask.astype(np.float32)
        return np.maximum(np.minimum(res, 1.), 0.) * 255.

    def load_train_data(self, batch_index):
        train_data = []
        ground_data = []
        isp_data = []
        for i in batch_index:
            _, name_train = os.path.split(self.training_train_file[i])
            name_train = name_train[:-3]
            name = name_train[:-5]+'.'

            train_img = np.expand_dims(np.load(os.path.join(self.training_data_path,
                                             self.subfolder_trainingdata, name_train + 'npy')), axis=2)
            ground_img = np.load(os.path.join(self.training_data_path,
                                              self.subfolder_groundtruth, name
                                              + 'npy'))
            isp_img = imread(os.path.join(self.training_data_path,
                                          self.subfolder_isp, name_train+'jpeg')).astype(np.float32)/255.
            h, w, _ = train_img.shape

            hi = random.choice(range(0, h-self.crop_size, 2))
            wi = random.choice(range(0, w-self.crop_size, 2))
            train_img = train_img[hi:hi+self.crop_size, wi:wi+self.crop_size]
            ground_img = ground_img[2*hi:2*(hi+self.crop_size), 2*wi:2*(wi+self.crop_size)]
            isp_img = isp_img[hi:hi+self.crop_size, wi:wi+self.crop_size]

            train_data.append(train_img)
            ground_data.append(ground_img)
            isp_data.append(isp_img)
        return np.stack(train_data), np.stack(ground_data), np.stack(isp_data)

    def write_test(self, epoch, **kwargs):
        epoch = str(epoch)
        path = os.path.join(self.result_path, epoch, self.test_image_folder)
        record = open(os.path.join(path, self.file_report), 'a+')
        for key in ['name', 'testing_sad_loss', 'testing_psnr_loss']:
            if key in kwargs and key=='name':
                record.write('name: %s\t' % kwargs[key])
            elif key in kwargs and key == 'testing_sad_loss':
                record.write('testing_sad_loss: %.6f\t' % float(kwargs[key]))
            elif key in kwargs and key == 'testing_psnr_loss':
                record.write('testing_psnr_loss: %.6f\t' % float(kwargs[key]))
        record.write('\n')
        record.close()

    def write(self, epoch, **kwargs):
        epoch = str(epoch)
        record = open(os.path.join(self.result_path, self.file_report), 'a+')
        record.write('epoch_num: %s\t' % epoch)
        keys = kwargs.keys()
        if 'training_sad_loss' in keys:
            record.write('train_sad_loss: %.6f\t' % float(kwargs['training_sad_loss']))
        if 'training_psnr_loss' in keys:
            record.write('train_psnr_loss: %.6f\t' % float(kwargs['training_psnr_loss']))
        if 'training_ssim_loss' in keys:
            record.write('train_ssim_loss: %.6f\t' % float(kwargs['training_ssim_loss']))
        if 'testing_sad_loss' in keys:
            record.write('test_sad_loss: %.6f\t' % float(kwargs['testing_sad_loss']))
        if 'testing_psnr_loss' in keys:
            record.write('test_psnr_loss: %.6f\t' % float(kwargs['testing_psnr_loss']))
        if 'testing_ssim_loss' in keys:
            record.write('test_ssim_loss: %.6f\t' % float(kwargs['testing_ssim_loss']))
        record.write('\n')
        record.close()

    def imgsave(self, img, epoch, name):
        epoch = str(epoch)
        if not os.path.isdir(os.path.join(self.result_path, epoch, self.test_image_folder)):
            if not os.path.isdir(os.path.join(self.result_path, epoch)):
                os.mkdir(os.path.join(self.result_path, epoch))
            os.mkdir(os.path.join(self.result_path, epoch, self.test_image_folder))
        if len(img.shape) == 4:
            img = img.reshape([img.shape[1], img.shape[2], img.shape[3]])
        imsave(os.path.join(self.result_path, epoch, self.test_image_folder, name+'png'), np.uint8(img))



