import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, unseen_classes, split_type='train', use_features=True, low_shot=True, percentage=0.01):
        self.data_path = data_path
        self.unseen_classes = unseen_classes
        self.split_type = split_type
        self.use_features = use_features
        self.low_shot = low_shot
        self.percentage = percentage
        if self.use_features:
            self.load_feature()
        else:
            self.load_data()
        if self.split_type == 'train' and self.low_shot == True:
            self.sample_data()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split_type == 'train':
            # read all training samples
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            # split seen and unseen classes
            unseen_samples_index_list = []
            for label_index, label_ele in enumerate(self.label):
                if label_ele in self.unseen_classes:
                    unseen_samples_index_list.append(label_index)
            self.data = np.delete(self.data, unseen_samples_index_list, axis=0)
            self.label = np.delete(self.label, unseen_samples_index_list, axis=0)
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split_type == 'test_zsl':
            # read all training samples
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            # ZSL setting - split seen and unseen classes
            unseen_samples_index_list = []
            for label_index, label_ele in enumerate(self.label):
                if label_ele in self.unseen_classes:
                    unseen_samples_index_list.append(label_index)
            self.data = self.data[unseen_samples_index_list]
            self.label = self.label[unseen_samples_index_list]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split_type == 'test_gzsl':
            # read all training samples
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            # GZSL setting
            self.data = self.data
            self.label = self.label
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    
    def load_feature(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split_type == 'train':
            # read all training samples
            self.data = npz_data['train_data']
            self.label = npz_data['train_label'] 
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split_type == 'test_zsl':
            # read all zsl samples
            self.data = npz_data['zsl_data']
            self.label = npz_data['zsl_label']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split_type == 'test_gzsl':
            # read all training samples
            self.data = npz_data['gzsl_data']
            self.label = npz_data['gzsl_label']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
    
    def sample_data(self):
        sampled_data = []
        sampled_label = []
        for cls in np.unique(self.label):
            cls_indices = np.where(self.label == cls)[0]
            sample_size = max(1, int(self.percentage * len(cls_indices)))  
            sampled_indices = np.random.choice(cls_indices, size=sample_size, replace=False)
            sampled_data.append(self.data[sampled_indices])
            sampled_label.append(self.label[sampled_indices])
        self.data = np.concatenate(sampled_data, axis=0)
        self.label = np.concatenate(sampled_label, axis=0)
        self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        print(f"Sampled {len(self.data)} samples for training.")

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        # for contrastive samples
        different_label_indices = [i for i in range(len(self.label)) if self.label[i] != label]
        if not different_label_indices:
            same_label_indices = [i for i in range(len(self.label)) if i != index and self.label[i] == label]
            if not same_label_indices:  
                contrast_index = index
            else:
                contrast_index = np.random.choice(same_label_indices)
        else:
            contrast_index = np.random.choice(different_label_indices)
        contrast_data_numpy = self.data[contrast_index]
        contrast_label = self.label[contrast_index]
        contrast_data_numpy = np.array(contrast_data_numpy)
        return data_numpy, label, contrast_data_numpy, contrast_label, index
    

