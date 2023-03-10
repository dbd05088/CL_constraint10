# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
from methods.er_baseline import ER
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from collections import Counter

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, ASERMemory
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i
# TODO 
# 지금은 Memory Retrieval만 다룸
# 하지만, Memory buffer replace하는 부분도 aser score 기반으로 바꿔야함

class ASER(ER):
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.memory = ASERMemory(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia, memory_size=self.memory_size)
        self.aser_type = kwargs["aser_type"]
        self.k = kwargs["k"]
        self.n_smp_cls = int(kwargs["n_smp_cls"])
        self.candidate_size = kwargs["aser_cands"]


    def mini_batch_deep_features(self, total_x, num):
        """
            Compute deep features with mini-batches.
                Args:
                    model (object): neural network.
                    total_x (tensor): data tensor.
                    num (int): number of data.
                Returns
                    deep_features (tensor): deep feature representation of data tensor.
        """
        
        with torch.no_grad():
            bs = 64
            num_itr = num // bs + int(num % bs > 0)
            sid = 0
            deep_features_list = []
            for i in range(num_itr):
                eid = sid + bs if i != num_itr - 1 else num
                batch_x = total_x[sid: eid]
                _, batch_deep_features_ = self.model(batch_x, get_feature=True)
                
                self.total_flops += (len(batch_x) * self.forward_flops)   
                deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
                sid = eid
                
            if num_itr == 1:
                deep_features_ = deep_features_list[0]
            else:
                deep_features_ = torch.cat(deep_features_list, 0)
                
        return deep_features_

    def deep_features(self, eval_x, n_eval, cand_x, n_cand):
        """
            Compute deep features of evaluation and candidate data.
                Args:
                    model (object): neural network.
                    eval_x (tensor): evaluation data tensor.
                    n_eval (int): number of evaluation data.
                    cand_x (tensor): candidate data tensor.
                    n_cand (int): number of candidate data.
                Returns
                    eval_df (tensor): deep features of evaluation data.
                    cand_df (tensor): deep features of evaluation data.
        """
        # Get deep features
        if cand_x is None:
            num = n_eval
            total_x = eval_x
        else:
            num = n_eval + n_cand
            total_x = torch.cat((eval_x, cand_x), 0)

        # compute deep features with mini-batches
        total_x = total_x.to(self.device)
        print("len total_x", len(total_x))
        deep_features_ = self.mini_batch_deep_features(total_x, num)

        eval_df = deep_features_[0:n_eval]
        cand_df = deep_features_[n_eval:]
        return eval_df, cand_df

    def sorted_cand_ind(self, eval_df, cand_df, n_eval, n_cand):
        """
            Sort indices of candidate data according to
                their Euclidean distance to each evaluation data in deep feature space.
                Args:
                    eval_df (tensor): deep features of evaluation data.
                    cand_df (tensor): deep features of evaluation data.
                    n_eval (int): number of evaluation data.
                    n_cand (int): number of candidate data.
                Returns
                    sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
        """
        # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
        # Preprocess feature vectors to facilitate vector-wise distance computation
        eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
        cand_df_tile = cand_df.repeat([n_eval, 1])
        # Compute distance between evaluation and candidate feature vectors
        distance_vector = (eval_df_repeat - cand_df_tile).pow(2).sum(1)
        # Turn distance vector into distance matrix
        distance_matrix = distance_vector.reshape((n_eval, n_cand))
        # Sort candidate set indices based on distance
        sorted_cand_ind_ = distance_matrix.argsort(1)
        return sorted_cand_ind_

    def compute_knn_sv(self, eval_x, eval_y, cand_x, cand_y, k):
        # Compute KNN SV score for candidate samples w.r.t. evaluation samples
        n_eval = eval_x.size(0)
        n_cand = cand_x.size(0)
        
        # Initialize SV matrix to matrix of -1
        sv_matrix = torch.zeros((n_eval, n_cand), device=self.device)
        # Get deep features
        eval_df, cand_df = self.deep_features(eval_x, n_eval, cand_x, n_cand)
        # Sort indices based on distance in deep feature space
        sorted_ind_mat = self.sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

        # Evaluation set labels
        el = eval_y
        el_vec = el.repeat([n_cand, 1]).T
        # Sorted candidate set labels
        cl = cand_y[sorted_ind_mat]

        # Indicator function matrix
        indicator = (el_vec == cl).float()
        indicator_next = torch.zeros_like(indicator, device=self.device)
        indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
        indicator_diff = indicator - indicator_next

        cand_ind = torch.arange(n_cand, dtype=torch.float, device=self.device) + 1
        denom_factor = cand_ind.clone()
        denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
        numer_factor = cand_ind.clone()
        numer_factor[k:n_cand - 1] = k
        numer_factor[n_cand - 1] = 1
        factor = numer_factor / denom_factor

        indicator_factor = indicator_diff * factor
        indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

        # Row indices
        row_ind = torch.arange(n_eval, device=self.device)
        row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

        # Compute SV recursively
        sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

        return sv_matrix
    

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        
        if len(sample) > 0:
            self.memory.register_stream(sample)

        stream_batch_size = len(sample)
        batch_size = min(batch_size, stream_batch_size + len(self.memory.images))
        memory_batch_size = batch_size - stream_batch_size
        for i in range(iterations):
            
            # for what to retrive from memory (Use Adversarial Shape Value)
            if memory_batch_size > 0:
                # 원래 aser은 buffer 가득 차면 시작임 따라서 최소한 original cifar10 buffer 크기인 500은 넘기도록
                if len(self.memory.images) >= 10 * self.candidate_size:
                    current_data, candidate_data, eval_data = self.memory.get_aser_calculate_batches(self.n_smp_cls, candidate_size=self.candidate_size) #memory_batch_size

                    eval_adv_x = current_data['image'].to(self.device)
                    eval_adv_y = current_data['label'].to(self.device)
                    cand_x = candidate_data['image'].to(self.device)
                    cand_y = candidate_data['label'].to(self.device)
                    cand_index = candidate_data['index'].to(self.device)
                    eval_coop_x = eval_data['image'].to(self.device)
                    eval_coop_y = eval_data['label'].to(self.device)
                    
                    self.model.eval()
                    sv_matrix_adv = self.compute_knn_sv(eval_adv_x, eval_adv_y, cand_x, cand_y, self.k)
                    sv_matrix_coop = self.compute_knn_sv(eval_coop_x, eval_coop_y, cand_x, cand_y, self.k)
                    
                    if self.aser_type == "asv":
                        # Use extremal SVs for computation
                        sv = sv_matrix_coop.max(0).values - sv_matrix_adv.min(0).values
                    else:
                        # Use mean variation for aser_type == "asvm" or anything else
                        sv = sv_matrix_coop.mean(0) - sv_matrix_adv.mean(0)
                        
                    ret_ind = sv.argsort(descending=True)
                    #ret_x = cand_x[ret_ind][:memory_batch_size]
                    #ret_y = cand_y[ret_ind][:memory_batch_size]
                    self.memory.register_batch_indices(batch_indices = cand_index[ret_ind][:memory_batch_size])

                else: # Random memory retrieval 
                    self.memory.register_batch_indices(batch_size = memory_batch_size)
                
                #data = self.memory.get_batch(batch_size, stream_batch_size)
                data = self.memory.get_aser_train_batches()
                
            else: # just Current Stream만 받는 것
                data = self.memory.get_batch(batch_size, stream_batch_size)
            
            self.model.train()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            
            # std check 위해서
            class_std, sample_std = self.memory.get_std()
            self.class_std_list.append(class_std)
            #self.sample_std_list.append(sample_std)

            
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x,y)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            '''
            print("###print parameter###")
            for name, child in self.model.named_children():
                for param in child.parameters():
                    if name == 'fc':
                        print(name, param)
            '''

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            print("self.total_flops", self.total_flops)
            
        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                    self.total_flops += (len(logit) * 4) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                self.total_flops += (len(logit) * 4) / 10e9
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    loss = self.criterion(logit, y)
                    self.total_flops += (len(logit) * 2) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                loss = self.criterion(logit, y)
                self.total_flops += (len(logit) * 2) / 10e9

        return logit, loss


    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass




