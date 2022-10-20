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
net_arg.add_argument('--quality_level', type=int, default=0, choices=[1, 2, 3, 4, 5, 6, 7, 8]) # 0 means nothing for this variable-rate model. All quality levels are used.
net_arg.add_argument('--N', type=int, default=192, choices=[128, 192])
net_arg.add_argument('--M', type=int, default=320, choices=[192, 256, 320])

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
train_arg.add_argument('--max_step', type=int, default=1200000)
train_arg.add_argument('--lr_update_step', type=int, default=50000)
train_arg.add_argument('--lr', type=float, default=5e-5)
# train_arg.add_argument('--lambda_', type=float, default=0) # 0 means nothing for this variable-rate model. This.
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='/')

def get_config():
    config, unparsed = parser.parse_known_args()
    data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
