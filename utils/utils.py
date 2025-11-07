import numpy as np 
import torch
import torch.optim as optim
import logging
import os 
import sys
import random

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_data(Feeder, args, unseen_classes, seed_fun, batch_size, split_type='train', use_features=True, low_shot=False, percentage=0.1):
    data_loader = torch.utils.data.DataLoader(
        dataset=Feeder(args.data_path, unseen_classes, split_type, use_features, low_shot, percentage),
        batch_size=batch_size,
        shuffle=True if split_type == 'train' else False,
        num_workers=args.num_worker,
        drop_last=True if split_type == 'train' else False,
        worker_init_fn=seed_fun)
    return data_loader


def task_definition(task):
    # task: ntu60_xsub_seen55_unseen5  or ntu60_xsub_seen55_unseen5_sadave_split1
    if task.count('_') == 3:
        task_name = task.split('_')[0] + '_' + task.split('_')[2] + '_' + task.split('_')[3]
    else:
        task_name = task.split('_')[0] + '_' + task.split('_')[2] + '_' + task.split('_')[3] + '_' + task.split('_')[4] + '_' + task.split('_')[5]

    if task_name == 'ntu60_seen55_unseen5':
        num_classes = 60
        unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
    elif task_name == 'ntu60_seen48_unseen12':
        num_classes = 60
        unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
    elif task_name == 'ntu60_seen40_unseen20':
        num_classes = 60
        unseen_classes = [0, 12, 13, 14, 15, 16, 17, 22, 23, 26, 29, 30, 31, 35, 36, 42, 43, 48, 56, 57]  # ntu60_40/20_split
    elif task_name == 'ntu60_seen30_unseen30':
        num_classes = 60
        unseen_classes = [0, 1, 2, 6, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 25, 26, 27, 31, 32, 33, 39, 42, 45, 47, 48, 51, 52, 55, 58, 59]  # ntu60_30/30_split
    elif task_name == 'ntu120_seen110_unseen10':
        num_classes = 120
        unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
    elif task_name == 'ntu120_seen96_unseen24':
        num_classes = 120
        unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split
    elif task_name == 'ntu120_seen80_unseen40':
        num_classes = 120
        unseen_classes = [11, 12, 18, 22, 23, 26, 28, 34, 37, 38, 42, 44, 46, 47, 48, 57, 59, 64, 66, 70, 73, 74, 75, 83, 86, 90, 92, 93, 95, 96, 102, 104, 107, 108, 110, 112, 115, 116, 118, 119]  # ntu120_80/40_split
    elif task_name == 'ntu120_seen60_unseen60':
        num_classes = 120
        unseen_classes = [0, 1, 4, 6, 7, 8, 9, 17, 18, 21, 23, 25, 26, 28, 30, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 50, 51, 52, 53, 56, 61, 62, 65, 67, 68, 69, 70, 74, 77, 78, 81, 83, 87, 89, 90, 91, 92, 94, 95, 96, 97, 100, 101, 109, 111, 114, 115, 116, 118]  # ntu120_60/60_split
    elif task_name == 'ntu60_seen55_unseen5_starsmie_split1':
        num_classes = 60
        unseen_classes = [4,19,31,47,51]   # split1: star-smie & shiftgcn
    elif task_name == 'ntu60_seen55_unseen5_starsmie_split2':
        num_classes = 60
        unseen_classes = [12,29,32,44,59]   # split2: star-smie & shiftgcn
    elif task_name == 'ntu60_seen55_unseen5_starsmie_split3':
        num_classes = 60
        unseen_classes = [7,20,28,39,58]   # split3: star-smie & shiftgcn
    elif task_name == 'ntu60_seen55_unseen5_sadave_split1':
        num_classes = 60
        unseen_classes = [0, 8, 15, 28, 46]   # split1 :sadave & stgcn
    elif task_name == 'ntu60_seen55_unseen5_sadave_split2':
        num_classes = 60
        unseen_classes = [15, 19, 23, 47, 50]   # split2 :sadave & stgcn
    elif task_name == 'ntu60_seen55_unseen5_sadave_split3':
        num_classes = 60
        unseen_classes = [29, 37, 38, 45, 55]   # split3 :sadave & stgcn
    elif task_name == 'ntu120_seen110_unseen10_starsmie_split1':
        num_classes = 120
        unseen_classes =  [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]   # split1: star-smie & shiftgcn
    elif task_name == 'ntu120_seen110_unseen10_starsmie_split2':
        num_classes = 120
        unseen_classes = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]   # split2: star-smie & shiftgcn
    elif task_name == 'ntu120_seen110_unseen10_starsmie_split3':
        num_classes = 120
        unseen_classes = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]   # split3: star-smie & shiftgcn
    elif task_name == 'ntu120_seen110_unseen10_sadave_split1':
        num_classes = 120
        unseen_classes = [0, 4, 6, 7, 24, 37, 54, 59, 97, 113]   # split1 :sadave & stgcn
    elif task_name == 'ntu120_seen110_unseen10_sadave_split2':
        num_classes = 120
        unseen_classes = [63, 79, 86, 92, 98, 100, 103, 110, 111, 117]   # split2 :sadave & stgcn
    elif task_name == 'ntu120_seen110_unseen10_sadave_split3':
        num_classes = 120
        unseen_classes = [9, 14, 17, 44, 60, 75, 81, 89, 108, 110]   # split3 :sadave & stgcn
    elif task_name == 'pku51_seen46_unseen5':
        num_classes = 51
        unseen_classes = [1, 9, 20, 34, 50]   # pku51_seen46_unseen5
    elif task_name == 'pku51_seen39_unseen12':
        num_classes = 51
        unseen_classes = [3,7,11,15,19,21,25,31,33,36,43,48]  # pku51_seen39_unseen12
    elif task_name == 'pku51_seen46_unseen5_sadave_split1':
        num_classes = 51
        unseen_classes = [10, 19, 27, 38, 48]   # split1 :sadave & stgcn
    elif task_name == 'pku51_seen46_unseen5_sadave_split2':
        num_classes = 51
        unseen_classes = [0, 9, 17, 30, 42]   # split2 :sadave & stgcn
    elif task_name == 'pku51_seen46_unseen5_sadave_split3':
        num_classes = 51
        unseen_classes = [18, 24, 31, 43, 45]   # split3 :sadave & stgcn
    elif task_name == 'pku51_seen46_unseen5_starsmie_split1':
        num_classes = 51
        unseen_classes = [3,14,29,31,49]  # split1: star-smie & shiftgcn
    elif task_name == 'pku51_seen46_unseen5_starsmie_split2':
        num_classes = 51
        unseen_classes = [2,15,39,41,43]  # split2: star-smie & shiftgcn
    elif task_name == 'pku51_seen46_unseen5_starsmie_split3':
        num_classes = 51
        unseen_classes = [4,12,16,22,36]  # split3: star-smie & shiftgcn
    else:
        raise NotImplementedError('Seen and unseen split errors!')
    seen_classes = list(set(range(num_classes))-set(unseen_classes))  # ntu60
    train_label_dict = {}
    for idx, l in enumerate(seen_classes):
        tmp = [0] * len(seen_classes)
        tmp[idx] = 1
        train_label_dict[l] = tmp
    test_zsl_label_dict = {}
    for idx, l in enumerate(unseen_classes):
        tmp = [0] * len(unseen_classes)
        tmp[idx] = 1
        test_zsl_label_dict[l] = tmp
    test_gzsl_label_dict = {}
    for idx, l in enumerate(range(num_classes)):
        tmp = [0] * num_classes
        tmp[idx] = 1
        test_gzsl_label_dict[l] = tmp
    return num_classes, unseen_classes, seen_classes, train_label_dict, test_zsl_label_dict, test_gzsl_label_dict

