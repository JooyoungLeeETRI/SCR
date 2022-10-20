from __future__ import print_function

import tensorflow_compression as tfc
# import sys
# import csv
import os
import math
import time

# from io import StringIO
# import scipy.special
import numpy as np
# from glob import glob
from tqdm import trange
# from itertools import chain
# from collections import deque

import scipy.misc
from models import *
from utils import save_image
from utils import save_recon_image
from msssim import MultiScaleSSIM

from PIL import Image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.clip_by_value((norm + 1) * 127.5, 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Manager(object):
    def __init__(self, config, data_loader_for_train):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.gpu_no = config.gpu_no
        self.global_step = tf.train.get_or_create_global_step()
        self.inc = tf.assign_add(self.global_step, 1, name='increment')

        self.config = config
        self.data_loader_for_train = data_loader_for_train
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.is_train = config.is_train
        self.is_test = config.is_test
        self.use_trainset = self.is_train

        self.lr = config.lr
        self.lr_new = tf.placeholder(shape=[], dtype=tf.float32)

        # self.lambda_base = config.lambda_

        self.model_dir = config.model_dir
        self.data_format = config.data_format

        if self.use_trainset:
            _, height, width, self.channel = \
                get_conv_shape(self.data_loader_for_train, self.data_format)
        else:
             self.channel = 3

        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.N = config.N
        self.M = config.M
        self.is_train = config.is_train

        if self.use_trainset:
            self.build_model_for_train()
        else:
            self.build_model_for_test()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        if self.use_trainset:
            save_sec = 300
        else:
            save_sec = 0

        variables_restored = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='HE') + \
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='HD') + \
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='E') + \
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D') + \
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='entropy_bottleneck')
        self.saver_for_loading_pretrained = tf.train.Saver(var_list=variables_restored)

        self.saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=save_sec,
                                 global_step=self.global_step,
                                 ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.is_test == True:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        else:
            if os.path.exists('{}/checkpoint'.format(self.model_dir)):
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
            else:
                print("config.pretrained_load_path:")
                print(config.pretrained_load_path)
                self.saver_for_loading_pretrained.restore(self.sess,
                                               tf.train.latest_checkpoint(config.pretrained_load_path))

        if not (self.use_trainset):
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

    def train(self):
        lr = self.lr
        start_step = 0
        for step in trange(start_step, self.max_step):
            if self.is_train:
                fetch_dict = {
                    "train_op": self.train_op,
                }

            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "total_loss": self.total_loss,
                })

            result = self.sess.run(fetch_dict,
                                   feed_dict={self.lr_new: lr})
            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                total_loss = result['total_loss']

                print("[{}/{}] total_loss: {:.6f}". \
                      format(step, self.max_step, total_loss))

            if step > self.max_step - 200000 and step <= self.max_step - 100000:
                lr = 1e-5
            elif step > self.max_step - 100000:
                lr = 2e-6

    def build_model_for_train(self):
        self.input_x = self.data_loader_for_train
        input_x = norm_img(self.input_x)

        with tf.device(self.gpu_no):
            self.entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
            self.y, encoder_variables = Encoder(input_x, self.N, self.M, self.data_format)
            self.z, HE_variables = Hyper_Encoder(self.y, self.N, self.data_format)

            self.z_hat, _ = self.entropy_bottleneck(self.z, training=False)
            self.z_tilde, self.z_likelihoods = self.entropy_bottleneck(self.z, training=True)

            self.pred_sigma, self.importance_map, _, _ = Hyper_Decoder(self.z_tilde, self.N, self.M, self.data_format)

            with tf.variable_scope("MASK") as vs:
                self.gamma = tf.maximum(tf.Variable(tf.ones([8, self.M]), name="GAMMA"), 0)

            with tf.variable_scope("ADAPTIVE_Q") as vs:
                TINY = 1e-5
                self.QV = tf.maximum(tf.Variable(tf.ones([8, self.M], dtype=tf.float32)), 0) + TINY
                self.IQV = tf.maximum(tf.Variable(tf.ones([8, self.M], dtype=tf.float32)), 0) + TINY

            SCALES_MIN = 0.11
            SCALES_MAX = 256
            SCALES_LEVELS = 64
            scale_table = np.exp(np.linspace(
                np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

            lambda_list = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
            self.total_loss = []
            for idx in range(8):
                gamma_q = self.gamma[idx]
                self.adjusted_importance_map = tf.math.pow(self.importance_map, gamma_q)

                samples = tf.random_uniform(shape=tf.shape(self.adjusted_importance_map), minval=0, maxval=1) + (
                        self.adjusted_importance_map - 0.5)
                g = tf.get_default_graph()
                with g.gradient_override_map({"Round": "Identity"}):
                    self.mask_relaxed = tf.round(samples)


                conditional_bottleneck = tfc.GaussianConditional(self.pred_sigma / self.QV[idx], scale_table,
                                                                 dtype=tf.float32)
                self.y_hat, _ = conditional_bottleneck(self.y / self.QV[idx], training=False)
                self.y_tilde, y_likelihoods = conditional_bottleneck(self.y / self.QV[idx], training=True)
                self.y_hat = self.y_hat * self.IQV[idx]
                self.y_tilde = self.y_tilde * self.IQV[idx]

                self.combined_y_tilde = self.mask_relaxed * self.y_tilde

                recon_image, decoder_variables = Decoder(self.combined_y_tilde, self.channel, self.N, self.M, self.data_format)

                G = recon_image
                self.recon_image = denorm_img(G)

                self.log_y_likelihoods = tf.log(y_likelihoods) * self.mask_relaxed
                self.log_z_likelihoods = tf.log(self.z_likelihoods)
                self.ch_info_amount = tf.reduce_sum(self.log_y_likelihoods, axis=[0, 1, 2]) / tf.reduce_sum(
                    self.log_y_likelihoods)

                self.recon_loss = tf.losses.mean_squared_error(input_x, G)
                self.recon_loss *= 127.5 ** 2

                NPIXEL = self.config.block_size ** 2
                self.estimated_y_bpp = tf.reduce_sum(self.log_y_likelihoods) / (-np.log(2) * NPIXEL * self.batch_size)
                self.estimated_z_bpp = tf.reduce_sum(self.log_z_likelihoods) / (-np.log(2) * NPIXEL * self.batch_size)

                self.estimated_bpp = self.estimated_y_bpp + self.estimated_z_bpp
                self.total_loss.append(lambda_list[idx] * self.recon_loss + self.estimated_bpp)

            self.total_loss = tf.reduce_sum(tf.convert_to_tensor(self.total_loss))

            if self.optimizer == 'adam':
                optimizer_fn = tf.train.AdamOptimizer
            else:
                raise Exception(
                    "[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.config.optimizer))
            optimizer = optimizer_fn(self.lr_new)
            grads_var = optimizer.compute_gradients(self.total_loss)
            grads = [g for g, _ in grads_var]
            grad_cond = tf.reduce_sum(grads[0])
            def f1():
                return True  # dummy fn
            def f2():
                return optimizer.apply_gradients(grads_var)

            main_step = tf.cond(tf.logical_or(tf.is_nan(grad_cond), tf.is_inf(grad_cond)), f1, f2)
            # main_step = optimizer.minimize(self.total_loss)

            aux_optimizer = optimizer_fn(self.lr_new * 10)
            aux_step = aux_optimizer.minimize(self.entropy_bottleneck.losses[0])

            self.train_op = tf.group(main_step, aux_step, self.entropy_bottleneck.updates[0], self.inc)

            with tf.device('/cpu:0'):
                self.summary_op = tf.summary.merge([
                    tf.summary.scalar("loss/recon_loss", self.recon_loss),
                    tf.summary.scalar("loss/estimated_bpp", self.estimated_bpp),
                    tf.summary.scalar("loss/total_loss", self.total_loss),
                    tf.summary.scalar("misc/lr", self.lr),
                ])

    def build_model_for_test(self):
        ###############################################################################################
        # Note: For the test phase, actual scaling using QV_q and IQV_q is performed in the test_encode_decode() function.
        # Here, we just prepare the tensor graph for performing test through the test_encode_decode() fuction.
        # So, please refer to the test_encode_decode() function to see how the models works in the test phase.
        ###############################################################################################
        self.input_x = tf.placeholder(
            shape=[1, None, None, self.channel], dtype=tf.float32)
        input_x = norm_img(self.input_x)

        with tf.device(self.gpu_no):
            self.entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
            self.y, encoder_variables = Encoder(input_x, self.N, self.M, self.data_format)
            self.z, HE_variables = Hyper_Encoder(self.y, self.N, self.data_format)
            self.z_hat, _ = self.entropy_bottleneck(self.z, training=False)
        with tf.device('/device:CPU:0'):
            self.pred_sigma, self.importance_map, _, _ = Hyper_Decoder(
                self.z_hat, self.N, self.M, self.data_format)

        with tf.variable_scope("MASK") as vs:
            self.gamma = tf.maximum(tf.Variable(tf.ones([8, self.M]), name="GAMMA"), 0)

        with tf.variable_scope("ADAPTIVE_Q") as vs:
            TINY = 1e-5
            self.QV = tf.maximum(tf.Variable(tf.ones([8, self.M], dtype=tf.float32)), 0) + TINY
            self.IQV = tf.maximum(tf.Variable(tf.ones([8, self.M], dtype=tf.float32)), 0) + TINY

        with tf.device(self.gpu_no):
            self.quality_level = tf.placeholder(dtype=tf.int32, name="Q_LEVEL")
            self.QV_q = tf.gather_nd(self.QV, [self.quality_level])
            self.IQV_q = tf.gather_nd(self.IQV, [self.quality_level])

            self.gamma_q = tf.gather_nd(self.gamma, [self.quality_level])
            self.adjusted_importance_map = tf.math.pow(self.importance_map, self.gamma_q)
            self.mask = tf.cast(tf.round(self.adjusted_importance_map), tf.bool)

            SCALES_MIN = 0.11
            SCALES_MAX = 256
            SCALES_LEVELS = 64
            scale_table = np.exp(np.linspace(
                np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

            self.sigma_input = tf.placeholder(tf.float32, [1, None])
            self.masked_y_input = tf.placeholder(tf.float32, [1, None])

            conditional_bottleneck = tfc.GaussianConditional(self.sigma_input, scale_table,
                                                             dtype=tf.float32)
            self.masked_y_hat, _ = conditional_bottleneck(self.masked_y_input, training=False)

            self.string_output = conditional_bottleneck.compress(self.masked_y_hat)
            self.string_input = tf.placeholder(tf.string, [1])
            self.symbols_output = conditional_bottleneck.decompress(self.string_input)

            self.z_in = self.z
            self.z_string_output = self.entropy_bottleneck.compress(self.z_in)
            self.z_string_input = tf.placeholder(tf.string, [1])

            self.z_shape_in = tf.placeholder(tf.int32, [2])
            z_shape = tf.concat([self.z_shape_in, [self.N]], axis=0)
            self.z_hat_output = self.entropy_bottleneck.decompress(
                self.z_string_input, z_shape, channels=self.N)

            self.combined_y_hat = tf.placeholder(
                shape=[1, None, None, self.M], dtype=tf.float32)
            recon_image, decoder_variables = Decoder(self.combined_y_hat, self.channel, self.N, self.M,
                                                     self.data_format)
            self.recon_image = denorm_img(recon_image)

    def get_patches(self, filepath):
        img = Image.open(filepath)
        w, h = img.size
        patches = []
        if self.config.grayscale:
            if img.mode == "RGB":
                img = np.dot(img, [0.2989, 0.5870, 0.1140]).reshape(w, h, 1)
            elif img.mode == "RGBA":
                img = np.dot(img, [0.2989, 0.5870, 0.1140, 0]).reshape(w, h, 1)
        patches.append(np.asarray(img))
        if self.data_format == 'NCHW':
            patches = np.transpose(patches, [0, 3, 1, 2])
        return np.asarray(patches), 1, 1

    def psnr(self, img1, img2):

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def get_scaling_vectors(self, quality_level):
        if quality_level - (quality_level // 1) == 0:
            quality_level = int(quality_level)
            QV_q, IQV_q = self.sess.run([self.QV_q, self.IQV_q],
                                                             feed_dict={self.quality_level: quality_level - 1})
        else:
            low_quality_level = quality_level // 1
            high_quality_level = low_quality_level + 1
            decimal_part = quality_level - low_quality_level

            low_QV_q, low_IQV_q = self.sess.run([self.QV_q, self.IQV_q],
                                                                     feed_dict={
                                                                         self.quality_level: low_quality_level - 1})
            high_QV_q, high_IQV_q = self.sess.run([self.QV_q, self.IQV_q],
                                                                       feed_dict={
                                                                           self.quality_level: high_quality_level - 1})
            QV_q = low_QV_q ** (1 - decimal_part) * high_QV_q ** decimal_part
            IQV_q = low_IQV_q ** (1 - decimal_part) * high_IQV_q ** decimal_part
        return QV_q, IQV_q

    def get_mask(self, importance_map, quality_level):
        if quality_level - (quality_level // 1) == 0:
            quality_level = int(quality_level)
            mask = self.sess.run(self.mask, feed_dict={self.importance_map: importance_map,
                                                       self.quality_level: quality_level - 1})
        else:
            low_quality_level = quality_level // 1
            high_quality_level = low_quality_level + 1
            decimal_part = quality_level - low_quality_level
            gamma_low = self.sess.run(self.gamma_q, feed_dict={self.quality_level: low_quality_level - 1})
            gamma_high = self.sess.run(self.gamma_q, feed_dict={self.quality_level: high_quality_level - 1})
            gamma_q = gamma_low ** (1 - decimal_part) * gamma_high ** decimal_part
            mask = self.sess.run(self.mask, feed_dict={self.importance_map: importance_map, self.gamma_q: gamma_q})
        return mask

    def test_encode_decode(self, filepath, file_idx, quality_level):
        os.makedirs('{}/{}'.format(self.model_dir, str(file_idx)), exist_ok=True)
        input_x, number_of_blocks_w, number_of_blocks_h = self.get_patches(filepath)
        h = input_x.shape[1]
        w = input_x.shape[2]

        path = self.model_dir + "/" + str(file_idx)
        dec_inputfile = enc_outputfile = path + "/y.bin"
        dec_hyperinputfile = enc_hyperoutputfile = path + "/z.bin"

        QV_q, IQV_q = self.get_scaling_vectors(quality_level)

        ############### encode z ####################################
        print("encoding started")
        start_time = time.time()
        y, z, z_hat = self.sess.run([self.y, self.z, self.z_hat], feed_dict={self.input_x: input_x})  # NCHW
        y = y / QV_q
        string = self.sess.run(self.z_string_output, feed_dict={self.z_in: z})

        with open(enc_hyperoutputfile, "wb") as file:
            file.write(string[0])

        file_size_in_bits = os.path.getsize(enc_hyperoutputfile) * 8
        bpp = file_size_in_bits / (y.shape[1] * y.shape[2] * 16 * 16)

        ############### encode y ####################################
        pred_sigma, importance_map = self.sess.run([self.pred_sigma, self.importance_map],
                                   feed_dict={self.z_hat: z_hat, self.quality_level: quality_level - 1})
        pred_sigma = pred_sigma / QV_q
        mask = self.get_mask(importance_map, quality_level)

        sigma_input = pred_sigma[mask]
        y_input = y[mask]
        string = self.sess.run(self.string_output,
                               feed_dict={self.sigma_input: np.expand_dims(sigma_input, 0),
                                          self.masked_y_input: np.expand_dims(y_input, 0)})

        time_encoding = time.time() - start_time

        with open(enc_outputfile, "wb") as file:
            file.write(string[0])


        print("encoding finished")
        file_size_in_bits = os.path.getsize(enc_outputfile) * 8
        bpp += file_size_in_bits / (y.shape[1] * y.shape[2] * 16 * 16)

        ############### decode z ####################################
        string = []
        with open(dec_hyperinputfile, "rb") as file:
            string.append(file.read())

        start_time = time.time()
        z_hat = self.sess.run(self.z_hat_output,
                              feed_dict={self.z_shape_in: [int(h / 64), int(w / 64)], self.z_string_input: string})
        z_entropy_decoding_time = time.time() - start_time

        start_time = time.time()
        pred_sigma, importance_map = self.sess.run([self.pred_sigma, self.importance_map],
                                   feed_dict={self.z_hat: z_hat, self.quality_level: quality_level - 1})
        pred_sigma = pred_sigma / QV_q
        mask = self.get_mask(importance_map, quality_level)
        hyper_dec_net_time = time.time() - start_time

        string = []
        with open(dec_inputfile, "rb") as file:
            string.append(file.read())

        ############### decode y ####################################
        start_time = time.time()
        mask_flat = mask.ravel()
        idx = np.nonzero(mask_flat)
        pred_sigma_flat = pred_sigma.ravel()
        pred_sigma_input = np.take(pred_sigma_flat, idx)
        overhead_time = time.time() - start_time

        start_time = time.time()
        y_hat_flat = self.sess.run(self.symbols_output,
                                      feed_dict={self.sigma_input: pred_sigma_input, self.string_input: string})
        y_hat_entropy_decoding_time = time.time() - start_time

        start_time = time.time()
        combined_y_hat = np.zeros_like(pred_sigma)
        combined_y_hat[mask] = y_hat_flat[0]  # combined_y_hat
        overhead_time += time.time() - start_time


        start_time = time.time()
        recon = self.sess.run(self.recon_image, {self.combined_y_hat: combined_y_hat * IQV_q})
        dec_net_time = time.time() - start_time

        print("*******************************")
        recon_img_path = '{}/{}/{}_recon.png'.format(self.model_dir, str(file_idx), quality_level)
        save_recon_image(recon,
                         recon_img_path,
                         number_of_blocks_w, number_of_blocks_h)

        print("{} saved".format(recon_img_path))
        print("time_encoding_time:{}".format(time_encoding))
        print("z_entropy_decoding_time:{}".format(z_entropy_decoding_time))
        print("hyper_dec_net_time:{}".format(hyper_dec_net_time))
        print("y_hat_entropy_decoding_time:{}".format(y_hat_entropy_decoding_time))
        print("dec_net_time:{}".format(dec_net_time))
        print("overhead_time:{}".format(overhead_time))
        print("mask TRUE ratio:{}".format(np.sum(mask) / (mask.size)))

        time_results = {
            "time_encoding": time_encoding,
            "z_entropy_decoding_time": z_entropy_decoding_time,
            "hyper_dec_net_time": hyper_dec_net_time,
            "y_hat_entropy_decoding_time": y_hat_entropy_decoding_time,
            "dec_net_time": dec_net_time,
            "overhead_time": overhead_time,
        }

        img1 = scipy.misc.imread(filepath, flatten=False, mode='RGB').astype(np.float32)
        img2 = scipy.misc.imread(recon_img_path, flatten=False, mode='RGB').astype(np.float32)

        print("bpp: " + str(bpp))

        psnr = self.psnr(img1, img2)
        print("psnr: " + str(psnr))

        img1 = np.reshape(img1, [-1, img1.shape[0], img1.shape[1], img1.shape[2]])
        img2 = np.reshape(img2, [-1, img2.shape[0], img2.shape[1], img2.shape[2]])

        ms_ssim = MultiScaleSSIM(img1, img2)
        print("ms-ssim: " + str(ms_ssim))
        print("*******************************")
        print("")

        return psnr, ms_ssim, bpp, time_results