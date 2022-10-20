#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', 1)

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# GPU
gpu_arg = add_argument_group('gpu')
gpu_arg.add_argument('--gpu_no', type=str, default='/device:GPU:0')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--quality_level', type=int, default=0, choices=[1, 2, 3, 4, 5, 6, 7, 8])
net_arg.add_argument('--N', type=int, default=128, choices=[128, 192])
net_arg.add_argument('--M', type=int, default=192, choices=[192, 256, 320])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--input_dataset', type=str, default='/workspace/datasets/CLIC/patches_256/')
data_arg.add_argument('--testset_path', type=str, default='./images/') #For test
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
# data_arg.add_argument('--num_worker', type=int, default=4)
data_arg.add_argument('--block_size', type=int, default=256) #patch block size

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=False)
train_arg.add_argument('--is_test', type=str2bool, default=False)

train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=7000000)
train_arg.add_argument('--lr_update_step', type=int, default=50000)
train_arg.add_argument('--lr', type=float, default=5e-5)
train_arg.add_argument('--lambda_', type=float, default=0)#####################
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='/')  ##############################################
# misc_arg.add_argument('--test_data_path', type=str, default=None,
#                       help='directory with images which will be used in test sample generation')
# misc_arg.add_argument('--sample_per_image', type=int, default=64,
#                       help='# of sample per image during test sample generation')
# misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
    config, unparsed = parser.parse_known_args()
    if config.quality_level in [1, 2, 3, 4, 5]:
        setattr(config, 'N', 128)
        setattr(config, 'M', 192)
    elif config.quality_level in [6, 7, 8]:
        setattr(config, 'N', 192)
        setattr(config, 'M', 320)
    else:
        print("input quality_level")
        exit()
    lambda_list = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
    setattr(config, 'lambda_', lambda_list[config.quality_level-1])

    data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
