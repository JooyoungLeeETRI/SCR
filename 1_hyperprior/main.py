import glob
import numpy as np
import tensorflow as tf
from manager import Manager
from config import get_config
from data_loader import get_loader
from utils import save_config
from PIL import Image
import argparse

def main(config):
    if config.lambda_ is not None:
        config.model_dir = 'logs/' + str(config.lambda_)

    if config.is_train:
        input_data_path = config.input_dataset
        data_loader_for_train = get_loader(
            config, input_data_path, config.batch_size, config.data_format, None, config.grayscale, False,
            path_block_size=config.block_size)
        manager = Manager(config, data_loader_for_train)
        save_config(config)
        manager.train()

    if config.is_test:
        file_idx=1
        file_list = glob.glob(config.testset_path + str("*.png"))
        psnr_list = []
        ms_ssim_list = []
        bpp_list = []

        manager = Manager(config, None)
        for filepath in file_list:
            psnr, ms_ssim, bpp, _ = manager.test_encode_decode(filepath, file_idx)
            psnr_list.append(psnr)
            ms_ssim_list.append(ms_ssim)
            bpp_list.append(bpp)
            file_idx = file_idx+1

        avg_psnr = np.mean(np.asarray(psnr_list))
        psnr_list.append(avg_psnr)
        avg_ms_ssim = np.mean(np.asarray(ms_ssim_list))
        ms_ssim_list.append(avg_ms_ssim)
        avg_bpp = np.mean(np.asarray(bpp_list))
        bpp_list.append(avg_bpp)

        np.savetxt('{}/TEST_RESULT_PSNR.csv'.format(config.model_dir), psnr_list, delimiter=",")
        np.savetxt('{}/TEST_RESULT_MS_SSIM.csv'.format(config.model_dir), ms_ssim_list, delimiter=",")
        np.savetxt('{}/TEST_RESULT_bpp.csv'.format(config.model_dir), bpp_list, delimiter=",")
if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
