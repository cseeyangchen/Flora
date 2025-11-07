import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])
from feeders import tools






class Feeder(Dataset):
    def __init__(self, data_path, unseen_classes, split_type='train',
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.unseen_classes = unseen_classes
        self.split_type = split_type
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()


    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        # read data
        if self.split_type == 'train':
            # read all training samples
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split_type == 'test':
            # read all testing samples
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('data split only supports train/test')
        # split seen and unseen classes
        unseen_samples_index_list = []
        for label_index, label_ele in enumerate(self.label):
            if label_ele in self.unseen_classes:
                unseen_samples_index_list.append(label_index)
        self.data = np.delete(self.data, unseen_samples_index_list, axis=0)
        self.label = np.delete(self.label, unseen_samples_index_list, axis=0)
        self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        # refine labels
        seen_classes = sorted(list(set(self.label)))
        label_dict = {}
        for idx, l in enumerate(seen_classes):
            label_dict[l] = idx
        for label_index, label_ele in enumerate(self.label):
            self.label[label_index] = label_dict[label_ele.item()]
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    


class Feeder_Extractor(Dataset):
    def __init__(self, data_path, unseen_classes, split_type='train',
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.unseen_classes = unseen_classes
        self.split_type = split_type
        # self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        # if normalization:
        #     self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        # read data
        if self.split_type == 'train_data':
            # read all training samples
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split_type == 'test_zsl' or self.split_type == 'test_gzsl':
            # read all testing samples
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('data split only supports train/test')
        # split data
        if self.split_type == 'train_data' or self.split_type == 'test_zsl':
            # split seen and unseen classes
            unseen_samples_index_list = []
            for label_index, label_ele in enumerate(self.label):
                if label_ele in self.unseen_classes:
                    unseen_samples_index_list.append(label_index)
            if self.split_type == 'train_data':
                self.data = np.delete(self.data, unseen_samples_index_list, axis=0)
                self.label = np.delete(self.label, unseen_samples_index_list, axis=0)
                self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            else:
                self.data = self.data[unseen_samples_index_list]
                self.label = self.label[unseen_samples_index_list]
                self.sample_name = ['test_zsl_' + str(i) for i in range(len(self.data))]
        # deal data
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

        

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)




