from model import model
import parameters
import tensorflow as tf
import numpy as np
from tools import tools
import os


class train:
    def __init__(self):
        self.real = parameters.REAL
        self.training = parameters.TRAINING
        self.testing = parameters.TESTING
        self.training_capacity = parameters.TRAINING_CAPACITY
        self.batch_size = parameters.BATCH_SIZE
        self.batch_per_epoch = parameters.BATCH_PER_EPOCH
        self.epoch_num = parameters.EPOCH_NUM
        self.pretrained = parameters.PRETRAINED
        self.crop_size = parameters.CROP_SIZE
        self.switch_epoch = parameters.SWITCH_EPOCH
        self.decay_coef = parameters.DECAY_COEF
        self.result_path = parameters.RESULT_PATH
        self.max_epoch = parameters.MAX_EPOCH
        self.learning_rate = parameters.LEARNING_RATE
        self.switch_learning_rate = parameters.SWITCH_LEARNING_RATE
        self.test_ratio = parameters.TEST_RATIO
        self.testing_capacity = parameters.TESTING_CAPACITY
        self.save_freq = parameters.SAVE_FREQ
        self.test_image_folder = parameters.TEST_IMAGE_FOLDER
        self.log_dir = parameters.LOG_DIR

    def help_eval(self, gt_img, res_img, sess):
        h, w, _ = res_img.shape
        gt_img_pd = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3])
        res_img_pd = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3])
        l1_loss = tf.reduce_mean(tf.abs(gt_img_pd-res_img_pd))
        psnr = tf.reduce_mean(tf.image.psnr(gt_img_pd, res_img_pd, max_val=1.))
        ssim = tf.reduce_mean(tf.image.ssim(gt_img_pd, res_img_pd, max_val=1.))
        feed_dict={gt_img_pd:gt_img.reshape([1, h, w, 3])/255., res_img_pd:res_img.reshape([1, h, w, 3])/255.}
        l1, p, s = sess.run([l1_loss, psnr, ssim], feed_dict=feed_dict)
        return l1, p, s

    def train(self):
        with tf.Graph().as_default():
            input_isp = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.crop_size, self.crop_size, 3],
                                       name='input_isp')
            input_img = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.crop_size, self.crop_size, 1],
                                       name='input_img')
            gt_img = tf.placeholder(dtype=tf.float32,
                                    shape=[self.batch_size, parameters.Scale * self.crop_size,
                                           parameters.Scale * self.crop_size, 3],
                                    name='gt_img')
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            mymodel = model(input_img=input_img, input_isp=input_isp, gt_img=gt_img, reuse=None)
            mytool = tools()
            loss, psnr, ssim, res = mymodel.build_model()
            index_queue = tf.train.range_input_producer(limit=self.training_capacity, shuffle=True)
            queue_op = index_queue.dequeue_many(self.batch_size)
            step = tf.Variable(0, trainable=False)
            training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss, global_step=step)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=tf.get_default_graph())
            saver = tf.train.Saver()

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)

            if self.pretrained:
                # sess.run(tf.global_variables_initializer())
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(self.log_dir))
                print('Finish loading pretrained model!')
            else:
                sess.run(tf.global_variables_initializer())

            for num_epoch in range(self.epoch_num, self.max_epoch + 1):
                train_loss = []
                train_psnr = []
                train_ssim = []
                if self.training:
                    for batch_num in range(0, int(self.batch_per_epoch)):
                        batch_index = sess.run(queue_op)
                        train_data, ground_data, isp_data = mytool.load_train_data(batch_index)

                        l_r = max(self.learning_rate * pow(self.decay_coef, num_epoch), self.switch_learning_rate) if num_epoch < self.switch_epoch else self.switch_learning_rate
                        print(l_r)

                        feed_dict = {input_img: train_data, input_isp: isp_data, gt_img: ground_data,
                                     learning_rate: l_r}
                        _, l, p, s, ss, r = sess.run([training_op, loss, psnr, ssim, step, res],
                                                              feed_dict=feed_dict)
                        print('epoch: %d, batch: %d, l1 loss: %.6f, psnr: %.6f, ssim: %.6f' % (
                        num_epoch, batch_num, l, p, s))

                        train_loss.append(l)
                        train_psnr.append(p)
                        train_ssim.append(s)
                    saver.save(sess=sess, save_path=self.log_dir + '/model.ckpt', global_step=step)

                if num_epoch % self.test_ratio == 0 and self.testing and not self.real:
                    print('Come to test')
                    test_loss = []
                    test_psnr = []
                    test_ssim = []

                    test_input_img = tf.placeholder(dtype=tf.float32, shape=[1, self.crop_size, self.crop_size, 1],
                                                    name='test_input_img')
                    test_input_isp = tf.placeholder(dtype=tf.float32, shape=[1, self.crop_size, self.crop_size, 3],
                                                    name='test_input_isp')

                    test_model = model(input_img=test_input_img, input_isp=test_input_isp, gt_img=None, reuse=True)
                    t_loss, t_psnr, t_ssim, t_res = test_model.build_model()

                    for i in range(self.testing_capacity):
                        print(i)

                        test_train_data, test_isp_data, h, w, test_name, gt_img_, mask = \
                            mytool.load_test_data(i)
                        res_list = []
                        for j in range(test_train_data.shape[0]):
                            feed_dict = {test_input_img: np.expand_dims(test_train_data[j, :, :, :], axis=0),
                                         test_input_isp: np.expand_dims(test_isp_data[j, :, :, :], axis=0)}

                            rt = sess.run([t_res], feed_dict=feed_dict)
                            rt = np.array(rt)
                            h_, w_, c = rt.shape[-3:]
                            rt = rt.reshape([h_, w_, c])
                            res_list.append(rt)
                        res_img = mytool.merge(np.stack(res_list), h, w, mask)
                        gtr_img = gt_img_*255.
                        l1, p, s = train.help_eval(self, gtr_img, res_img, sess)
                        test_loss.append(l1)
                        test_psnr.append(p)
                        test_ssim.append(s)
                        if num_epoch % self.save_freq == 0:
                            rt_img = res_img
                            if not os.path.isdir(os.path.join(self.result_path, str(num_epoch))):
                                os.mkdir(os.path.join(self.result_path, str(num_epoch)))
                                os.mkdir(os.path.join(self.result_path, str(num_epoch), self.test_image_folder))
                            mytool.imgsave(rt_img, num_epoch, test_name)
                            mytool.write_test(epoch=num_epoch, name=test_name, testing_sad_loss=l1,
                                              testing_psnr_loss=p)
                    mytool.write(epoch=num_epoch, training_sad_loss=np.mean(train_loss),
                                 training_psnr_loss=np.mean(train_psnr), training_ssim_loss=np.mean(train_ssim),
                                 testing_sad_loss=np.mean(test_loss),
                                 testing_psnr_loss=np.mean(test_psnr), testing_ssim_loss=np.mean(test_ssim))
                elif not self.real:
                    mytool.write(epoch=num_epoch, training_sad_loss=np.mean(train_loss),
                                 training_ssim_loss=np.mean(train_ssim),
                                 training_psnr_loss=np.mean(train_psnr))

                if self.real:
                    test_input_img = tf.placeholder(dtype=tf.float32, shape=[1, self.crop_size, self.crop_size, 1],
                                                    name='test_input_img')
                    test_input_isp = tf.placeholder(dtype=tf.float32, shape=[1, self.crop_size, self.crop_size, 3],
                                                    name='test_input_isp')

                    test_model = model(input_img=test_input_img, input_isp=test_input_isp, gt_img=None, reuse=True)
                    t_loss, t_psnr, t_ssim, t_res = test_model.build_model()

                    for i in range(self.testing_capacity):

                        test_train_data, test_isp_data, h, w, test_name, mask = \
                            mytool.load_test_real(i)
                        res_list = []
                        for j in range(test_train_data.shape[0]):
                            feed_dict = {test_input_img: np.expand_dims(test_train_data[j, :, :, :], axis=0),
                                         test_input_isp: np.expand_dims(test_isp_data[j, :, :, :], axis=0)}

                            rt = sess.run([t_res], feed_dict=feed_dict)
                            rt = np.array(rt)
                            h_, w_, c = rt.shape[-3:]
                            rt = rt.reshape([h_, w_, c])
                            res_list.append(rt)
                        res_img = mytool.merge(np.stack(res_list), h, w, mask)
                        rt_img = res_img
                        mytool.imgsave(rt_img, num_epoch, '{}_highresolution.'.format(test_name))
                if not self.training:
                    print('Validation finished!')
                    exit(0)


if __name__ == '__main__':
    mytrain = train()
    mytrain.train()
