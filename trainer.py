import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from models.fm import MappingNet
import utils.utils as utils

class Trainer:
    def __init__(self, args):
        self.args = args
        ##### ---- Logger ---- #####
        self.logger = utils.get_logger(self.args.out_dir)
        self.logger.info(f'Training Flora on {args.task_name} dataset.')
        ##### ---- Task ---- #####
        self.load_task()
        ##### ---- Load Data ---- #####
        self.load_data()
        ##### ---- Initialize MappingNet and Optimizer---- #####
        self.load_mapping_model()
        ##### ---- Initialize Seen Semantics & Skeleton Features ---- #####
        self.load_prototypes()
        ##### ---- Initialize Best Accuracy ---- #####
        self.best_accuracy = [float('-inf')]*4  # [ZSL GZSL-Seen GZSL-Unseen GZSL-H]

    
    def load_task(self):
        # load train and test data
        self.num_classes, self.unseen_classes, self.seen_classes, self.train_label_dict, self.test_zsl_label_dict, self.test_gzsl_label_dict = utils.task_definition(self.args.task_name)
        self.logger.info(f'Loading task done.')
    
    def load_data(self):
        Feeder = utils.import_class(self.args.feeder)
        self.data_loader = dict()
        self.data_loader['train'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.train_batch_size, split_type='train', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading Training dataloader done.')
        self.data_loader['test_zsl'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.test_batch_size, split_type='test_zsl', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading ZSL Testing dataloader done.')
        self.data_loader['test_gzsl'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.test_batch_size, split_type='test_gzsl', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading GZSL Testing dataloader done.')
        self.train_loader_iter = utils.cycle(self.data_loader['train'])
        self.zsl_loader  = self.data_loader['test_zsl']
        self.gzsl_loader = self.data_loader['test_gzsl']
    
    def load_mapping_model(self):
        self.mapping_model = MappingNet(args=self.args)
        self.logger.info(f'Loading mapping network done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self.optimizer_mapping = optim.AdamW(self.mapping_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.logger.info(f'Loading mapping network optimizer done.')
    
    def load_prototypes(self):
        if self.args.setting == 'ZSL':
            self.args.mixed_weights = self.args.mixed_weights[0]
        elif self.args.setting == 'GZSL':
            self.args.mixed_weights = self.args.mixed_weights[1]
        else:
            raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
        # load semantcis prototypes
        semantic_embeddings = self.mapping_model.class_embeddings  # (60, token, 768*2) 
        # load seen & unseen semantics similarity
        semantic_embeddings_mean = semantic_embeddings.mean(1)  # (60, 768*2)
        self.text_similarity = F.cosine_similarity(semantic_embeddings_mean.unsqueeze(1), semantic_embeddings_mean.unsqueeze(0), dim=-1)  # (60, 60)
        self.topk_similarity_indices = torch.topk(self.text_similarity, k=self.args.topk+1, dim=-1).indices[:, 1:]  # (60, topk) except itself
        self.topk_similarity_values = torch.topk(self.text_similarity, k=self.args.topk+1, dim=-1).values[:, 1:]  # (60, topk)  except itself
        # computing neighbor semantics
        self.attuned_semantics = torch.zeros(semantic_embeddings.size(), device=semantic_embeddings.device)  # seen+unseen token 768*2
        for k in range(self.args.topk):
            topi_semantics = torch.zeros(self.attuned_semantics.size(),device=semantic_embeddings.device)  # seen+unseen token 768*2
            for i in range(self.attuned_semantics.shape[0]):
                topi_semantics[i] = (self.topk_similarity_values[i, k] * semantic_embeddings[self.topk_similarity_indices[i, k], :, :]).unsqueeze(0)
            self.attuned_semantics += self.args.mixed_weights / self.args.topk * topi_semantics  # seen+unseen token 768*2
        self.attuned_semantics += semantic_embeddings # seen+unseen token 768*2
        self.logger.info(f'Attuned Semantics Successfully.')
    
    def train_learning_phase(self):
        ###### ---- Phase 1: Neighbor-aware Semantic Learning ---- #####
        avg_loss, avg_mu, avg_logvar, avg_recon = 0., 0., 0., 0.
        for nb_iter in range(1, self.args.align_iter+ 1):
            gt_motion, label_idx, _, _,  _ = next(self.train_loader_iter)
            gt_motion = gt_motion.cuda().float() # (bs, 2, 16, 512)
            label_idx = label_idx.cuda() # (bs, )
            true_label_array = torch.tensor([self.train_label_dict[l.item()] for l in label_idx]).cuda()
            # train
            metrics = self.mapping_model.forward_phase1(gt_motion, self.attuned_semantics, self.seen_classes, torch.argmax(true_label_array, dim=1))  # (bs, 2, 16, 512)
            # Combined loss
            loss = metrics['loss']
            loss_mu = metrics['loss_mu']
            loss_logvar = metrics['loss_log_var']
            loss_recon = metrics['loss_recon']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()
            avg_loss += loss.item()
            avg_mu += loss_mu.item()
            avg_logvar += loss_logvar.item()
            avg_recon += loss_recon.item()
            # print
            if nb_iter % self.args.print_iter ==  0 :
                avg_loss /= self.args.print_iter
                avg_mu /= self.args.print_iter
                avg_logvar /= self.args.print_iter
                avg_recon /= self.args.print_iter
                self.logger.info(f"Train Phase 1. Iter {nb_iter} : Loss. {avg_loss:.5f} \t Loss_mu. {avg_mu:.5f} \t Loss_logvar. {avg_logvar:.5f} \t Loss_Recon. {avg_recon:.5f}")
                avg_loss, avg_mu, avg_logvar, avg_recon = 0., 0., 0., 0.
        self.logger.info(f'Phase 1: Neighbor-aware Semantic Learning Done.')


    def train_deciding_phase(self):
        ###### ---- Phase 2: Open-form Flow Deciding ---- #####
        avg_loss, avg_fm, avg_fm_con = 0., 0., 0.
        for nb_iter in range(self.args.align_iter+ 1, self.args.total_iter+ 1):
            gt_motion, label_idx, gt_motion_contrastive, label_idx_contrastive, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.cuda().float() # (bs, 2, 16, 512)
            gt_motion_contrastive = gt_motion_contrastive.cuda().float() # (bs, 2, 16, 512)
            label_idx = label_idx.cuda() # (bs, )
            label_idx_contrastive = label_idx_contrastive.cuda() # (bs, )
            true_label_array = torch.tensor([self.train_label_dict[l.item()] for l in label_idx]).cuda()
            true_label_array_contrastive = torch.tensor([self.train_label_dict[l.item()] for l in label_idx_contrastive]).cuda()
            # train
            metrics = self.mapping_model.forward_phase2(gt_motion, gt_motion_contrastive, torch.argmax(true_label_array, dim=1), torch.argmax(true_label_array_contrastive, dim=1), self.seen_classes, self.attuned_semantics)  
            # Combined loss
            loss = metrics['loss'] 
            loss_fm = metrics['loss_flow_matching']
            loss_fm_con = metrics['loss_flow_matching_contrastive']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()
            avg_loss += loss.item()
            avg_fm += loss_fm.item()
            avg_fm_con += loss_fm_con.item()
            # print
            if nb_iter % self.args.print_iter ==  0 :
                avg_loss /= self.args.print_iter
                avg_fm /= self.args.print_iter
                avg_fm_con /= self.args.print_iter
                self.logger.info(f"Train Phase 3. Iter {nb_iter} : Loss. {avg_loss:.5f} \t Loss FM. {avg_fm:.5f} \t Loss FM Con. {avg_fm_con:.5f}")
                avg_loss, avg_fm, avg_fm_con = 0., 0., 0.

            # test
            if nb_iter % self.args.eval_iter==0:
                if self.args.setting == 'ZSL':
                    self.test_zsl(nb_iter)
                elif self.args.setting == 'GZSL':
                    self.test_gzsl(nb_iter)
                elif self.args.setting == 'All':
                    self.test_zsl(nb_iter)
                    self.test_gzsl(nb_iter)
                else:
                    raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
        
        # print best accuracy
        self.print_best_accuracy()


    def test_zsl(self, nb_iter):
        self.mapping_model.eval()
        total_samples = 0
        correct_predictions = 0
        for batch_idx, (gt_motion, label_idx, _, _, _) in enumerate(self.zsl_loader):
            with torch.no_grad():
                gt_motion = gt_motion.cuda().float() # (bs, 2, 16, 512)
                true_label_array = torch.tensor([self.test_zsl_label_dict[l.item()] for l in label_idx]).cuda()
                # calculate accuracy -- velocity error
                res_dis = self.mapping_model.predict_zsl(gt_motion, self.unseen_classes, self.attuned_semantics, torch.argmax(true_label_array, dim=1))
                labels = torch.argmax(true_label_array, dim=1)
                predictions = res_dis['predictions_idx']  # (bs,)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += gt_motion.size(0)
        accuracy_zsl = 100 * correct_predictions / total_samples
        # update best accuracy
        self.best_accuracy[0] = max(self.best_accuracy[0], accuracy_zsl)
        self.logger.info(f"Iter {nb_iter}: ZSL Accuracy: {accuracy_zsl:.2f}% ")
        self.mapping_model.train()
    
    def test_gzsl(self, nb_iter):
        self.mapping_model.eval()
        total_samples = 0
        correct_predictions = 0
        label_statistics = []
        pred_list = []
        for batch_idx, (gt_motion, label_idx, _, _, _) in enumerate(self.gzsl_loader):
            with torch.no_grad():
                gt_motion = gt_motion.cuda().float() # (bs, 2, 16, 512)
                true_label_array = torch.tensor([self.test_gzsl_label_dict[l.item()] for l in label_idx]).cuda()
                # calculate accuracy -- velocity error
                res_dis = self.mapping_model.predict_gzsl(gt_motion, self.seen_classes, self.unseen_classes, self.attuned_semantics)
                labels = torch.argmax(true_label_array, dim=1)
                predictions = res_dis['predictions_idx']  # (bs, num_noise)
                correct_predictions += (predictions == labels).sum().item()
                label_statistics.append(labels.cpu().numpy())  # 记录标签统计
                pred_list.append(predictions.cpu().numpy())  # (bs, n)
                total_samples += gt_motion.size(0)

        # store statistics 
        label_statistics = np.concatenate(label_statistics, axis=0)  # n
        pred_list = np.concatenate(pred_list, axis=0)  # n
        # gzsl
        seen_total_samples, unseen_total_samples = 0, 0
        seen_correct_samples, unseen_correct_samples = 0, 0
        for ture_label, pred_label in zip(label_statistics, pred_list):
            if ture_label in self.seen_classes:
                seen_total_samples += 1
                if ture_label == pred_label:
                    seen_correct_samples += 1
            else:
                unseen_total_samples += 1
                if ture_label == pred_label:
                    unseen_correct_samples += 1
        seen_accuracy = 100 * seen_correct_samples / seen_total_samples if seen_total_samples > 0 else 0
        unseen_accuracy = 100 * unseen_correct_samples / unseen_total_samples if unseen_total_samples > 0 else 0
        if (seen_accuracy + unseen_accuracy) > 0:
            harmonic_mean = 2 * (seen_accuracy * unseen_accuracy) / (seen_accuracy + unseen_accuracy)
        else:
            harmonic_mean = 0
        self.logger.info(f"Iter {nb_iter}: Seen Accuracy: {seen_accuracy:.2f}% \t Unseen Accuracy: {unseen_accuracy:.2f}% \t Harmonic Mean: {harmonic_mean:.2f}%")
        # store best model
        if harmonic_mean > self.best_accuracy[3]:
            # update best accuracy
            self.best_accuracy[1] = seen_accuracy
            self.best_accuracy[2] = unseen_accuracy
            self.best_accuracy[3] = harmonic_mean
        self.mapping_model.train()


    def print_best_accuracy(self):
        if self.args.setting == 'ZSL':
            self.logger.info(f'Best ZSL Results: {self.best_accuracy[0]:.2f}%')
        elif self.args.setting == 'GZSL':
            self.logger.info(f'Best GZSL Results: Seen: {self.best_accuracy[1]:.2f}% Unseen: {self.best_accuracy[2]:.2f}% H: {self.best_accuracy[3]:.2f}%')
        else:
            raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
    
    def train(self):
        self.train_learning_phase()
        self.train_deciding_phase()
        
                
        

    
    


   
   
    