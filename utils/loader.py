# -*- coding: utf-8 -*-

"""
Data loader for NYTFB-68K/NYTFB-280K data
"""

import os
import math
import gensim
import time
import sys
import json
import copy
import operator


import numpy as np
from scipy.misc import logsumexp
from random import shuffle

from scipy.sparse import hstack, vstack
from collections import defaultdict, Counter
from gensim.models import word2vec



class DataLoader(object):
    """
    Load data from txt files and preprocess them.
    """
    def __init__(self, opt, word_map, rel_map):
        
        train_file = opt['train_file']
        test_file = opt['test_file']
        self.opt = opt
        self.word_map = word_map
        self.rel_map = rel_map
        limit = opt['pos_limit']
        
        pos_min_e1, pos_max_e1 = 0, 0
        pos_min_e2, pos_max_e2 = 0, 0
        
        start_time = time.time()
        
        bags_train, train_rel, train_list, train_pos_e1, train_pos_e2, \
        pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2 = self.load_from_file(train_file, "###END###\n", word_map, rel_map, limit, \
                                                                        pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2)
        
        stop_time = time.time()   
        # print "Time to load training data: ", stop_time - start_time
        print "Training data loaded."
        
        start_time = time.time()
        
        bags_test, test_rel, test_list, test_pos_e1, test_pos_e2, \
        pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2 = self.load_from_file(test_file, "\t###END###\n", word_map, rel_map, limit, \
                                                                        pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2)
        
        
        stop_time = time.time()   
        # print "Time to load test data: ", stop_time - start_time
        print "Test data loaded."
        
        train_pos_e1, train_pos_e2, test_pos_e1, test_pos_e2 = self.pos_norm(train_pos_e1, train_pos_e2, test_pos_e1, test_pos_e2, pos_min_e1, pos_min_e2)
        
        
        # Generate label for each bag
        train_bags_label = self.get_label_each_bag(bags_train, train_rel, opt)
        test_bags_label = self.get_label_each_bag(bags_test, test_rel, opt)
        
        
        self.bags_train = bags_train
        self.train_rel = train_rel
        self.train_list = train_list
        self.train_pos_e1 = train_pos_e1
        self.train_pos_e2 = train_pos_e2
        self.train_bags_label = train_bags_label
        
        self.bags_test = bags_test
        self.test_rel = test_rel
        self.test_list = test_list
        self.test_pos_e1 = test_pos_e1
        self.test_pos_e2 = test_pos_e2
        self.test_bags_label = test_bags_label
        
        self.pos_min_e1 = pos_min_e1
        self.pos_max_e1 = pos_max_e1
        self.pos_min_e2 = pos_min_e2
        self.pos_max_e2 = pos_max_e2
        
        
        
    def load_from_file(self, file_name, end_str, word_map, rel_map, limit, \
                       pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2):
        
        """
        Each line is a relation mention
        """
        
        train_rel = [] # Contain id of relation for each sentence
        train_list = [] # Append the idxs of words for each sentence
        train_pos_e1 = [] # Append distances to the first entity
        train_pos_e2 = [] # Append distances to the second entity

        bags_train = defaultdict(list)

        with open(file_name,'rb') as f:
            for item in f:

                [e1, e2, head_s, tail_s, relation, sentence] = item.decode('utf-8').strip(end_str).split('\t')
        
                if relation in rel_map:
                    num = rel_map[relation]
                else:
                    num = 0
                
                length, lefnum, rignum = 0, 0, 0
                
                words = sentence.split()
                
                con = []
                
                for word in words:
                    
                    if word == head_s:
                        lefnum = length
                        
                    if word == tail_s:
                        rignum = length
                
                    if word not in word_map:
                        con.append(0)
                    else:   
                        con.append(word_map[word])
                        
                    length += 1
                
                # Cannot find the corresponding entity in the sentence
                if lefnum == 0 and rignum == 0:
        #             print head_s, '-', tail_s, '-'
        #             print sentence
                    continue
                  
                # If entity is the last word
                if lefnum == length-1 or rignum == length-1:
                    
                    # Add an '.' at the end
                    con.append(word_map['.'])
                    length += 1
                    
                # if with_rel: 
                #     bags_train[e1+u'\t'+e2+u'\t'+relation].append(len(train_rel))
                # else:
                bags_train[e1+u'\t'+e2].append(len(train_rel))
                    
                train_rel.append(num)
                
                conl = [lefnum - i for i in xrange(length)]
                conr = [rignum - i for i in xrange(length)]
                
                for i in xrange(length):
                    if conl[i] >= limit:
                        conl[i] = limit
                    elif conl[i] <= -limit:
                        conl[i] = -limit
                        
                    if conr[i] >= limit:
                        conr[i] = limit
                    elif conr[i] <= -limit:
                        conr[i] = -limit
                        
                if conl[0] > pos_max_e1: 
                    pos_max_e1 = conl[0]
                if conl[-1] < pos_min_e1:
                    pos_min_e1 = conl[-1]
                    
                if conr[0] > pos_max_e2:
                    pos_max_e2 = conr[0]
                if conr[-1] < pos_min_e2:
                    pos_min_e2 = conr[-1]
                    
                train_list.append(con) # Append the ids of words for each sentence
                train_pos_e1.append(conl) # Append distances to the first entity
                train_pos_e2.append(conr) # Append distances to the second entity
            

        # print pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2, len(bags_train)

        return bags_train, train_rel, train_list, train_pos_e1, train_pos_e2, \
               pos_min_e1, pos_max_e1, pos_min_e2, pos_max_e2
                
                
                
            
    def pos_norm(self, train_pos_e1, train_pos_e2, test_pos_e1, test_pos_e2, pos_min_e1, pos_min_e2):
        """
        Make the position of entity in the middle
        """
        
        # start_time = time.time()

        for i in xrange(len(train_pos_e1)):
            
            work1 = [item - pos_min_e1 for item in train_pos_e1[i]]
            train_pos_e1[i] = work1
            
            work2 = [item - pos_min_e2 for item in train_pos_e2[i]]
            train_pos_e2[i] = work2

        for i in xrange(len(test_pos_e1)):
            
            work1 = [item - pos_min_e1 for item in test_pos_e1[i]]
            test_pos_e1[i] = work1
            
            work2 = [item - pos_min_e2 for item in test_pos_e2[i]]
            test_pos_e2[i] = work2
            
        # stop_time = time.time()   
        # print "Time to update word position: ", stop_time - start_time

        return train_pos_e1, train_pos_e2, test_pos_e1, test_pos_e2
    
    
            
    def get_label_each_bag(self, bags_EP, rel_list, options):
        """
        Generate the label list for each EP bag
        """
        bags_rel = {}
        
        for bag_name in bags_EP.keys():
            
            bag_label = np.zeros(options['num_rel'], dtype=np.int)
            bag_label[[rel_list[sen_id] for sen_id in bags_EP[bag_name]]] = 1
            bags_rel[bag_name] = bag_label
        
        return bags_rel
        
        
        
        
        
