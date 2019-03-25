# -*- coding: utf-8 -*-

import os
import math
import gensim
import time
import sys
import json
import copy
import operator
import argparse
import utils
import random
from datetime import datetime

import numpy as np
from scipy.misc import logsumexp

from collections import defaultdict, Counter
from gensim.models import word2vec
from sklearn import metrics

from utils import loader, helper, kb_info
from model.PCNN_NMAR import PCNN_NMAR


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='saved_models')

    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=50, help='Word embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=5, help='Position embedding dimension.')
    parser.add_argument('--pos_limit', type=int, default=30, help='Position embedding length limit.')
    parser.add_argument('--num_conv', type=int, default=230, help='The number of convolutional filters.')
    parser.add_argument('--win_size', type=int, default=3, help='Convolutional filter size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='The rate at which randomly set a parameter to 0.')
    parser.add_argument('--lr', type=float, default=0.01, help='Applies to SGD.')
    parser.add_argument('--num_epoch', type=int, default=15)
    parser.add_argument('--num_rand_start', type=int, default=30)
    parser.add_argument('--penal_scalar', type=int, default=500)
    
    parser.add_argument('--adaplr', dest='adaplr', action='store_true', help='Use bag-size adaptive learning rate.')
    parser.add_argument('--no-adaplr', dest='adaplr', action='store_false')
    parser.set_defaults(adaplr=True)
    parser.add_argument('--adaplr_beta1', type=float, default=20.0)
    parser.add_argument('--adaplr_beta2', type=float, default=25.0)
    
    parser.add_argument('--sen_file', type=str, default='sentential_DEV.txt', help='Sentential eval dataset.')
    parser.add_argument('--heldout_eval', type=bool, default=False, help='Perform heldout evaluation after each epoch.')
    parser.add_argument('--save_each_epoch', type=bool, default=False, help='Save the checkpoint of each epoch.')

    # parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--trial_exp', dest='trial', action='store_true', help='Use partial training data.')
    parser.set_defaults(trial=False)
    parser.add_argument('--num_trial', type=int, default=10000)
    parser.add_argument('--log_step', type=int, default=20000)
    parser.add_argument('--num_exp', type=int, default=0)

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    
    if args.cpu:
        args.cuda = False
    
#     # Set random seed
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
     
        
    # make opt
    opt = vars(args)

    opt['train_file'] = opt['data_dir'] + '/' + 'train.txt'
    opt['test_file'] = opt['data_dir'] + '/' + 'test.txt'
    opt['sen_dev_file'] = opt['data_dir'] + '/' + 'sentential_DEV.txt'
    opt['vocab_file'] = opt['data_dir'] + '/' + 'vec.bin'
    opt['rel_file'] = opt['data_dir'] + '/' + 'relation2id.txt'
    if opt['data_dir'].split('/')[-1] != '':
        opt['data_name'] = opt['data_dir'].split('/')[-1]
    else:
        opt['data_name'] = opt['data_dir'].split('/')[-2]


    # Pretrained word embedding
    print "\nPretrained word embedding loaded"
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(opt['vocab_file'], binary=True)
    word_list = [u'UNK'] + w2v_model.index2word
    word_vec = w2v_model.syn0

    word2id = {}

    for id, word in enumerate(word_list):
        word2id[word] = id

    assert opt['emb_dim'] == w2v_model.syn0.shape[1]


    # Read from relation2id.txt to build a dictionary: rel2id
    rel2id = {}
            
    with open(opt['rel_file'],'rb') as f:
        for item in f:
            [relation, id] = item.strip('\n').split(' ')
            rel2id[relation] = int(id)
            
    id2rel = [''] * len(rel2id)
    for relation, rel_id in rel2id.items():
        id2rel[rel_id] = relation

    opt['num_rel'] = len(rel2id)
    opt['vocab_size'] = len(word_list)


    # Load data
    all_data = loader.DataLoader(opt, word2id, rel2id)
    opt['pos_e1_size'] = all_data.pos_max_e1 - all_data.pos_min_e1 + 1
    opt['pos_e2_size'] = all_data.pos_max_e2 - all_data.pos_min_e2 + 1
    opt['pos_min_e1'] = all_data.pos_min_e1
    opt['pos_min_e2'] = all_data.pos_min_e2
    opt['EP_num_train'] = len(all_data.bags_train)
    opt['EP_num_test'] = len(all_data.bags_test)
    

    assert opt['pos_e1_size'] == opt['pos_e2_size']

    helper.check_dir(opt['save_dir'])
    helper.print_config(opt)
    
    
    # Get KB disagreement penalty
    kb_score_all = kb_info.get_MIT_MID_score(all_data.bags_train, all_data.train_bags_label, opt, rel2id, id2rel)
    
    
    # Get hamming score
    ham_score_all = kb_info.getting_hamming_score(all_data.bags_train, all_data.train_bags_label, opt)
    
    
    # Build the model
    PCNN_NMAR_model = PCNN_NMAR(word_vec, opt)
    
    if opt['cuda']:
        PCNN_NMAR_model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(PCNN_NMAR_model.parameters(), lr=opt['lr'])
    
    print "Training starts."

    for epoch in xrange(opt['num_epoch']):
        
        opt['epoch'] = epoch
        
        start_time = time.time()
        
        total_loss = np.float64(0.0)
        
        train_part = all_data.bags_train.keys()[:]
            
        if opt['trial']:
            train_part = train_part[:opt['num_trial']]
            
        random.shuffle(train_part) 
        
        
        for index, bag_name in enumerate(train_part):
            
            if index > 0 and index % opt['log_step'] == 0:
                print '{}: train examples {}/{} (epoch {}/{}), loss = {:.6f} '.format(datetime.now(), index, opt['EP_num_train'], epoch+1, opt['num_epoch'], total_loss)
                
            optimizer.zero_grad()
                
            sentence_list = all_data.bags_train[bag_name]
            target = all_data.train_bags_label[bag_name]
            kb_score = kb_score_all[bag_name]
            ham_score = ham_score_all[bag_name]

            BPable_loss, loss_augmented = PCNN_NMAR_model(sentence_list, target, all_data, kb_score, ham_score)
            
            # Check if there is search error
            assert loss_augmented >= 0
            
            total_loss += loss_augmented
           
            
            # Apply bag-size adaptive learning rate
            if opt['adaplr']:
                if len(sentence_list) <= opt['adaplr_beta1']:
                    adaplr_scalar = 1
                elif len(sentence_list) <= opt['adaplr_beta2']:
                    adaplr_scalar = (float(opt['adaplr_beta1']) / len(sentence_list))
                else:
                    adaplr_scalar = (float(opt['adaplr_beta1']) / len(sentence_list)) ** 2
                    
                BPable_loss = BPable_loss * adaplr_scalar

            BPable_loss.backward()
            optimizer.step()

            
            
        stop_time = time.time()  
        print 'For epoch {}/{}, training time:{}, training loss: {:.6f}'.format(epoch+1, opt['num_epoch'], stop_time - start_time, total_loss)
        

            
        # Sentential evaluation
        sen_AUC = PCNN_NMAR_model.sentential_eval(opt['sen_dev_file'], all_data, rel2id, id2rel)
        print 'The sentential AUC of P/R curve on DEV set: {:.3f}'.format(sen_AUC)
        
        
        # Heldout evaluation
        if opt['heldout_eval']:
            recall, precision = PCNN_NMAR_model.heldout_eval(all_data)
            heldout_AUC = metrics.auc(recall, precision) if len(recall) != 0 else 0
            print "The heldout AUC of P/R curve: {:.4f}".format(heldout_AUC)
        
        
        # Save parameters in each epoch
        model_file = opt['save_dir'] + '/' + opt['data_name'] + '_' + \
                    'lr{}_penal{}_epoch{}.tar'.format(opt['lr'], opt['penal_scalar'], epoch)
        # print model_file
        
        if opt['save_each_epoch']:
            torch.save({
                'state_dict': PCNN_NMAR_model.state_dict(),
                'config': opt
            }, model_file )
            
        
        best_file = opt['save_dir'] + '/' + opt['data_name'] + '_' + \
                    'lr{}_penal{}_best_model.tar'.format(opt['lr'], opt['penal_scalar'])
        
        if epoch == 0 or best_AUC < sen_AUC:
            
            best_AUC = sen_AUC
            
            torch.save({
                'state_dict': PCNN_NMAR_model.state_dict(),
                'config': opt
            }, best_file )
          


if __name__ == "__main__":
    main()





    


