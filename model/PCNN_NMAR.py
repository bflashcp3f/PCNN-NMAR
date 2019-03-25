# -*- coding: utf-8 -*-

import os
import math
import gensim
import time
import sys
import json
import copy
import operator
import heapq
import utils
import inference

import numpy as np
from scipy.misc import logsumexp
from random import shuffle
from sklearn import metrics

from collections import defaultdict, Counter
from gensim.models import word2vec
from numpy import unravel_index


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd



# MODEL
class PCNN_NMAR(nn.Module):
    
    def __init__(self, pretrained_word_emb, opt):
        
        super(PCNN_NMAR, self).__init__()
        
        self.emb_dim = opt['emb_dim']
        self.pos_dim = opt['pos_dim']
        
        d_all = self.emb_dim + 2 * self.pos_dim
        
        self.win_size = opt['win_size']
        self.num_conv = opt['num_conv']

        self.pos_e1_size = opt['pos_e1_size']
        self.pos_e2_size = opt['pos_e2_size']
        self.pos_min_e1 = opt['pos_min_e1']
        self.pos_min_e2 = opt['pos_min_e2']

        self.dropout = opt['dropout']
        
        self.word_embedding = nn.Embedding(opt['vocab_size'], self.emb_dim)
        self.pos_embedding_e1 = nn.Embedding(opt['pos_e1_size'], self.pos_dim)
        self.pos_embedding_e2 = nn.Embedding(opt['pos_e2_size'], self.pos_dim)
        
        # if opt['cuda']:
        #     torch.cuda.manual_seed(opt['seed'])
        # else:
        #     torch.manual_seed(opt['seed'])
        
        self.conv1 = nn.Conv2d(1, self.num_conv, (self.win_size, d_all), padding=(self.win_size-1, 0))
        self.linear = nn.Linear(self.num_conv*3, opt['num_rel'])
        
        self.opt = opt
        self.param_initialization(pretrained_word_emb)
        
        
        
    def norm2(self, mat):
        v = torch.from_numpy(mat)
        v = F.normalize(v, p=2, dim=1)
        return v

        
    def param_initialization(self, pretrained_word_emb):
        
        
        # Word embedding is pretrained, and others are randomly initialized
        pretrained_word_emb = np.concatenate((np.random.uniform(-1, 1, size=(1, self.emb_dim)), pretrained_word_emb), axis=0)
        pos_emb1 = np.random.uniform(-1, 1, size=(self.pos_e1_size, self.pos_dim))
        pos_emb2 = np.random.uniform(-1, 1, size=(self.pos_e2_size, self.pos_dim))
        
        # Do p_2norm (following c++ code)
        pretrained_word_emb = self.norm2(pretrained_word_emb)
        pos_emb1 = self.norm2(pos_emb1)
        pos_emb2 = self.norm2(pos_emb2)
        
        # Copy data matrix to parameters
        self.word_embedding.weight.data.copy_(pretrained_word_emb)
        self.pos_embedding_e1.weight.data.copy_(pos_emb1)
        self.pos_embedding_e2.weight.data.copy_(pos_emb2)
        
        # Only matrix uses xavier_uniform, bias not
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.uniform_(self.conv1.bias)
        
    
          
    def pairwise_pooling(self, sen_em, e1, e2):
        
        # As we pad (win_size-1) at the start and end of the sentence,
        # there is a "+ self.win_size" when calculate the index
        e1_posi = np.where(e1 == -self.pos_min_e1)[0][0] + self.win_size 
        e2_posi = np.where(e2 == -self.pos_min_e2)[0][0] + self.win_size
        
        if e1_posi > e2_posi:
            e1_posi, e2_posi = e2_posi, e1_posi
        
        sen_em1 = sen_em[:, :, :e1_posi, :]
        sen_em2 = sen_em[:, :, e1_posi:e2_posi, :]
        
        try:
            sen_em3 = sen_em[:, :, e2_posi:, :]
        except:
            print e1_posi, e2_posi, sen_em.size()
            raise
        
        # pw_result is a (num_conv * 3) matrix
        pw_result = torch.cat([F.max_pool2d(sen_em1, kernel_size=sen_em1.size()[2:]).view(-1, 1), 
                                   F.max_pool2d(sen_em2, kernel_size=sen_em2.size()[2:]).view(-1, 1),
                                   F.max_pool2d(sen_em3, kernel_size=sen_em3.size()[2:]).view(-1, 1)], 1)

        return pw_result
    
    
    
    def generate_representation(self, sen_list, train_lists, train_position_e1, train_position_e2):
        
        for index, sentence in enumerate(sen_list):
            
            if self.opt['cuda']:
                sig_sentence = autograd.Variable( torch.LongTensor(train_lists[sentence]).cuda() )
                pos_e1 = autograd.Variable( torch.LongTensor(train_position_e1[sentence]).cuda() )
                pos_e2 = autograd.Variable( torch.LongTensor(train_position_e2[sentence]).cuda() )
            else:
                sig_sentence = autograd.Variable( torch.LongTensor(train_lists[sentence]) )
                pos_e1 = autograd.Variable( torch.LongTensor(train_position_e1[sentence]) )
                pos_e2 = autograd.Variable( torch.LongTensor(train_position_e2[sentence]) )
            
            words_embeds = self.word_embedding(sig_sentence)
            pos_e1_embeds = self.pos_embedding_e1(pos_e1)
            pos_e2_embeds = self.pos_embedding_e2(pos_e2)

            sentence_embeds = torch.cat([words_embeds, pos_e1_embeds, pos_e2_embeds], 1).unsqueeze(0).unsqueeze(0)

            sentence_embeds = self.conv1(sentence_embeds)
            
            # The result sentence_embeds is a (dim_C * 3) matrix
            sentence_embeds = self.pairwise_pooling(sentence_embeds, pos_e1.data.cpu().numpy(), pos_e2.data.cpu().numpy())
                
            sentence_embeds = torch.tanh(sentence_embeds).view(1, -1)

            if index == 0:
                bag_embeds = sentence_embeds
            else:
                bag_embeds = torch.cat([bag_embeds, sentence_embeds], 0)
    
        return bag_embeds
        
        
    def forward(self, sen_list, target, all_data, penal_data, penal_ham):
        
        train_lists = all_data.train_list
        train_position_e1 = all_data.train_pos_e1
        train_position_e2 = all_data.train_pos_e2
        
        ran_num = self.opt['num_rand_start']
        relation_num = self.opt['num_rel']
        
    
        bag_embeds = self.generate_representation(sen_list, train_lists, train_position_e1, train_position_e2)
        
        bag_embeds = F.dropout2d(bag_embeds, p=self.dropout, training=True)
        
        post_z = self.linear(bag_embeds)
        
        
        # MAP inference with KB
        z_star, rel_star = inference.local_search(post_z.data.cpu().numpy().astype(np.float), penal_data, \
                                                  ran_num, relation_num)
        
        # MAP inference without KB
        # mention-level hamming loss
        z_prim, rel_prim = inference.loss_augmented_search(post_z.data.cpu().numpy().astype(np.float), \
                                                           np.array(z_star), relation_num)
        hamming_loss = inference.get_hamming_loss(np.array(z_star), z_prim.astype(int), len(sen_list))
        
        
        # # relation-level hamming loss
        # z_prim, rel_prim = inference.local_search(post_z.data.cpu().numpy().astype(np.float), penal_ham,
        #                                           ran_num, relation_num)
        # hamming_loss = inference.get_hamming_loss(np.array(rel_star), rel_prim.astype(int), relation_num)
        
        
        
        if self.opt['cuda']:
            z_star = autograd.Variable( torch.LongTensor(z_star).cuda().long().view(-1, 1))
            z_prim = autograd.Variable( torch.LongTensor(z_prim).cuda().long().view(-1, 1))
        else:
            z_star = autograd.Variable( torch.LongTensor(z_star).long().view(-1, 1))
            z_prim = autograd.Variable( torch.LongTensor(z_prim).long().view(-1, 1))
        
        BPable_loss = post_z.gather(1, z_prim).sum() - post_z.gather(1, z_star).sum()
        
        return BPable_loss, BPable_loss.data.cpu().numpy() + hamming_loss
    
    
    def sentential_eval(self, sen_file, all_data, relation2id, id2relation):
        """
        Mention-level relation extraction on annotated Hoffman et. al.datase
        """
        
        sentential_labels = []
        
        with open(sen_file, 'r') as f:
            for item in f.readlines():

                [e1_id, e2_id, sen_index_in_bag, relation, manual_label, sentence_annoated, e1, e2, sentence] = item.decode('utf-8').strip('\n').split('\t')

                key = e1_id + "\t" + e2_id  
            
                # ignore relation /location/administrative_division/country
                # since it is just the inverse of 
                # /location/country/administrative_divisions which is also
                # in the dataset

                if relation == "/location/administrative_division/country":
                    continue

                assert relation in relation2id   
                assert key in all_data.bags_test
                    
                sen_list = all_data.bags_test[key]

                for index, sentence in enumerate(sen_list):
                    
                    if self.opt['cuda']:
                        sig_sentence = autograd.Variable( torch.LongTensor(all_data.test_list[sentence]).cuda() )
                        pos_e1 = autograd.Variable( torch.LongTensor(all_data.test_pos_e1[sentence]).cuda() )
                        pos_e2 = autograd.Variable( torch.LongTensor(all_data.test_pos_e2[sentence]).cuda() )
                    else:
                        sig_sentence = autograd.Variable( torch.LongTensor(all_data.test_list[sentence]) )
                        pos_e1 = autograd.Variable( torch.LongTensor(all_data.test_pos_e1[sentence]) )
                        pos_e2 = autograd.Variable( torch.LongTensor(all_data.test_pos_e2[sentence]) )

                    words_embeds = self.word_embedding(sig_sentence)
                    pos_e1_embeds = self.pos_embedding_e1(pos_e1)
                    pos_e2_embeds = self.pos_embedding_e2(pos_e2)

                    sentence_embeds = torch.cat([words_embeds, pos_e1_embeds, pos_e2_embeds], 1).unsqueeze(0).unsqueeze(0)

                    sentence_embeds = self.conv1(sentence_embeds)

                    sentence_embeds = self.pairwise_pooling(sentence_embeds, pos_e1.data.cpu().numpy(), pos_e2.data.cpu().numpy())

                    sentence_embeds = torch.tanh(sentence_embeds).view(1, -1)

                    if index == 0:
                        bag_embeds = sentence_embeds
                    else:
                        bag_embeds = torch.cat([bag_embeds, sentence_embeds], 0)

                # prob_matrix = F.softmax(0.5 * self.linear(bag_embeds), dim = 1).data.cpu().numpy()
                prob_matrix = F.softmax(self.linear(bag_embeds), dim = 1).data.cpu().numpy()

                label = Label()
                label.relation = relation
                label.tf = (manual_label =="y" or manual_label == "indirect")
                label.name1 = e1
                label.name2 = e2
                label.sentence = sentence
                label.pred_score = prob_matrix[int(sen_index_in_bag)][relation2id[relation]]              
                sentential_labels.append(label)
                
                
        # max recall
        MAX_TP = 0
        for l in sentential_labels:
            if l.relation != 'NA' and l.tf:
                MAX_TP += 1

        # sort predictions by decreasing score
        sentential_labels.sort(key=lambda x: x.pred_score, reverse=True)

        curve = []
        f1 = []
        TP = 0
        FP = 0
        FN = 0

        for l in sentential_labels:

            if l.tf:
                TP += 1
            else:
                FP += 1

            if TP + FP == 0:
                precision = 0.0
            else:
                precision = TP / float(TP + FP)
            recall = TP / float(MAX_TP)
            curve.append((precision, recall))

            if precision == 0 and recall == 0:
                f1_value = 0
            else:
                f1_value = 2 * (precision * recall) / (precision + recall)

            f1.append(f1_value)

        recall_list = [item[1] for item in curve]
        precision_list = [item[0] for item in curve]

        return metrics.auc(recall_list, precision_list)
    

    def heldout_eval(self, all_data):
        """
        EP-level automatic RE evaluation on NYTFB-68K/NYTFB-280K
        """
        prob_label_result = []
        

        for index, bag_name in enumerate(all_data.bags_test.keys()[:]):

            # if index > 0 and index % 20000 == 0:
            #     print 'index == ', index 
            
            z_prim = {}
            sen_list = all_data.bags_test[bag_name]
            
            for index, sentence in enumerate(sen_list):
                
                if self.opt['cuda']:
                    sig_sentence = autograd.Variable( torch.LongTensor(all_data.test_list[sentence]).cuda() )
                    pos_e1 = autograd.Variable( torch.LongTensor(all_data.test_pos_e1[sentence]).cuda() )
                    pos_e2 = autograd.Variable( torch.LongTensor(all_data.test_pos_e2[sentence]).cuda() )
                else:
                    sig_sentence = autograd.Variable( torch.LongTensor(all_data.test_list[sentence]) )
                    pos_e1 = autograd.Variable( torch.LongTensor(all_data.test_pos_e1[sentence]) )
                    pos_e2 = autograd.Variable( torch.LongTensor(all_data.test_pos_e2[sentence]) )
            
                words_embeds = self.word_embedding(sig_sentence)
                pos_e1_embeds = self.pos_embedding_e1(pos_e1)
                pos_e2_embeds = self.pos_embedding_e2(pos_e2)
                
                sentence_embeds = torch.cat([words_embeds, pos_e1_embeds, pos_e2_embeds], 1).unsqueeze(0).unsqueeze(0)
                
                sentence_embeds = self.conv1(sentence_embeds)
                
                sentence_embeds = self.pairwise_pooling(sentence_embeds, pos_e1.data.cpu().numpy(), pos_e2.data.cpu().numpy())
                
                sentence_embeds = torch.tanh(sentence_embeds).view(1, -1)

                sentence_prob = F.softmax(self.linear(sentence_embeds), dim = 1)
                
                value, pred_class = torch.max(sentence_prob, 1)
                
                val, p_class = value.data.cpu().numpy()[0], pred_class.data.cpu().numpy()[0]
                
                if p_class not in z_prim or val > z_prim[p_class]:
                    z_prim[p_class] = val
                    
            target = list(set([all_data.test_rel[sen] for sen in sen_list]))
                    
            for p_class, prob in z_prim.items():
                if p_class == 0:
                    continue
                if p_class in target:
                    prob_label_result.append((prob, p_class, 1))
                else:
                    prob_label_result.append((prob, p_class, 0))

        prob_label_result_sorted = sorted(prob_label_result, key=operator.itemgetter(0), reverse=True)

        TP = 0
        FP = 0
        all_pos = sum([len(set([all_data.test_rel[sen] for sen in sen_list if all_data.test_rel[sen] != 0])) for sen_list in all_data.bags_test.values()])

        precision = []
        recall = []
        f1 = []

        # for (prob, label, result) in prob_label_result_sorted[:2000]:
        for (prob, label, result) in prob_label_result_sorted:
            if result == 1:
                TP += 1
            else:
                FP += 1

            pre_value = float(TP)/(TP+FP)
            rec_value = float(TP)/all_pos
            if pre_value == 0 and rec_value == 0:
                f1_value = 0
            else:
                f1_value = 2 * (pre_value * rec_value) / (pre_value + rec_value)

            precision.append(pre_value)
            recall.append(rec_value)
            f1.append(f1_value)
            
        
        return recall, precision


class Label():
    
    def __init__(self):
        
        self.name1 = ''
        self.name2 = ''
        self.relation = 0
        self.sentence = ''
        self.tf = False
        self.pred_score = 0.0