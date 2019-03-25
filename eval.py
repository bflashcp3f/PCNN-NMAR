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
import pickle
import glob

import numpy as np
from scipy.misc import logsumexp
from random import shuffle

from scipy.sparse import hstack, vstack
from collections import defaultdict, Counter
from gensim.models import word2vec
from sklearn import metrics

from utils import loader, helper
from model.PCNN_NMAR import PCNN_NMAR


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='saved_models/', help='Directory of the model.')
    parser.add_argument('--model_name', type=str, default='best_model.tar', help='Name of the model file.')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

    parser.add_argument('--emb_dim', type=int, default=50, help='Word embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=5, help='Position embedding dimension.')
    parser.add_argument('--pos_limit', type=int, default=30, help='Position embedding length limit.')
    parser.add_argument('--num_conv', type=int, default=230, help='The number of convolutional filters.')
    parser.add_argument('--win_size', type=int, default=3, help='Convolutional filter size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='The rate at which randomly set a parameter to 0.')
    parser.add_argument('--lr', type=float, default=0.001, help='Applies to SGD.')
    parser.add_argument('--num_epoch', type=int, default=15)
    parser.add_argument('--seed', type=int, default=666)
    
    parser.add_argument('--sentential_eval', type=bool, default=False, help='Perform sentential evaluation.')
    parser.add_argument('--sen_file', type=str, default='', help='Sentential eval dataset.')
    
    parser.add_argument('--heldout_eval', type=bool, default=False, help='Perform heldout evaluation after each epoch.')
    parser.add_argument('--print_config', type=bool, default=False, help='Print out the configuration of the model.')
    
    parser.add_argument('--tune', type=bool, default=False, help='Perform sentential evaluation for all models in the same directory.')

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    # parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()

    if args.cpu:
        args.cuda = False
    
        
    # make opt
    opt = vars(args)

    opt['train_file'] = opt['data_dir'] + '/' + 'train.txt'
    opt['test_file'] = opt['data_dir'] + '/' + 'test.txt'
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

    if opt['tune']:
        model_file_list = sorted(glob.glob(args.model_dir + opt['data_name'] + "*.tar"))
    else:
        model_file_list = [args.model_dir + '/' + args.model_name]
    
    # model_file = args.model_dir + '/' + args.model_name
    
    for model_file in model_file_list:

        # Load input model
        print("Load model: {}".format(model_file.split('/')[-1]))
        PCNN_NMAR_model = PCNN_NMAR(word_vec, opt)
        checkpoint = torch.load(model_file)
        PCNN_NMAR_model.load_state_dict(checkpoint['state_dict'])
        # model_config = torch.load(model_file)['config']

        # if opt['print_config']:
            # helper.print_config(model_config)


        if opt['cuda']:
            PCNN_NMAR_model.cuda()

        # Sentential evaluation
        if opt['sentential_eval']:

            print "Sentential evaluaiton starts."

            sen_file = opt['data_dir'] + '/' + opt['sen_file']
            sen_AUC = PCNN_NMAR_model.sentential_eval(sen_file, all_data, rel2id, id2rel)
            print "The sentential AUC of P/R curve on {} is {:.3f}".format(opt['sen_file'], sen_AUC)

            print "Sentential evaluaiton ends.\n"

        # Heldout evaluation
        if opt['heldout_eval']:

            print "Heldout evaluation starts."

            recall, precision = PCNN_NMAR_model.heldout_eval(all_data)
            heldout_AUC = metrics.auc(recall, precision) if len(recall) != 0 else 0
            print "The heldout AUC of P/R curve is {:.4f}".format(heldout_AUC)

            print "Heldout evaluaiton ends."


if __name__ == "__main__":
    main()







