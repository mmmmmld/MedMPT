# !/usr/bin/env python
# -*-coding:utf-8 -*-
import argparse
import os
import datetime
import yaml
from easydict import EasyDict
import zipfile
import socket

def parse_config(config_file):
    with open(config_file, encoding="GBK") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config

def makedir(dir_path):
    if isinstance(dir_path, list):
        for path in dir_path:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def getTime():
    timeNow = datetime.datetime.now().strftime('%b%d_%H-%M')
    return str(timeNow)

def save_code(dir_path, zip_path):
    '''
    :param dir_path: directory to be zipped
    :param zip_path: path to save zipped file
    :return:
    '''

    zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(dir_path):
        file_path = root.replace(dir_path, '')
        exp_dir = False
        for exp_name in ['checkepoints', 'events', 'results']:
            if exp_name in file_path:
                exp_dir = True
                continue
        if exp_dir:
            continue
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.yaml'):
                zip.write(os.path.join(root, filename), os.path.join(file_path, filename))
    zip.close()

def Get_local_ip():
    """
    Returns the actual ip of the local machine.
    This code figures out what source address would be used if some traffic
    were to be sent out to some well known address on the Internet. In this
    case, a Google DNS server is used, but the specific address does not
    matter much. No traffic is actually sent.
    """
    try :
        csock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csock.connect(( '166.111.8.28' , 80 ))
        (addr, port) = csock.getsockname()
        csock.close()
        return addr
    except socket.error:
        return "127.0.0.1"

def str2bool(v):
    if v.lower() in ['true', 'ture', 'yes', 'y', 't', '1']:
        return True
    elif v.lower() in ['false', 'no', 'n', 'f', '0']:
        return False
    elif v.lower() in ['none', ]:
        return None
    else:
        raise argparse.ArgumentTypeError("Unsupport value encountered.")

def str2list(v):
    # input must be a string like: 1,2,3,
    if len(v) == 0 or v == '[]':
        return []
    return list(eval(v))

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
