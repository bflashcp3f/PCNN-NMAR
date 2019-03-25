# -*- coding: utf-8 -*-
import numpy as np

from collections import Counter


def get_entity_frequencty(train_data):
    """
    Count the frequency of each entity in the training data
    """
    entities_in_train = []
    
    for item in train_data.keys():
        entities_in_train += item.split('\t')
        
    return Counter(entities_in_train)


def get_MIT_MID_score(data, relation_data, opt, relation2id, id2relation):
    """
    Get a penalty-score vetor for each bag
    """
    penal_vec_all = {}
    
    # id2relation = [''] * opt['num_rel']
    # for relation, rel_id in relation2id.items():
    #     id2relation[rel_id] = relation
        
    entity_freq = get_entity_frequencty(data)
    
    for bag_name in data.keys():
        penal_vec = np.zeros(opt['num_rel'])
        
        [e1_id, e2_id] = bag_name.split('\t')
        
        # bag_label = list(set([relation_data[sen] for sen in data[bag_name]]))
        bag_label = relation_data[bag_name]
        
        for rel_id in range(opt['num_rel']):
            
            if id2relation[rel_id] == 'NA':
                penal_vec[rel_id] = -1
            # elif rel_id not in bag_label:
            elif not bag_label[rel_id]:
                penal_vec[rel_id] = -5
            else:
                # Two sets of penalty scores
                if opt['data_name'] == 'NYTFB-280K':
                    if id2relation[rel_id] in ["/location/location/contains", "/people/person/place_lived",
                                               "/people/person/nationality", "/business/person/company",
                                               "/people/person/place_of_birth"]:
                        penal_vec[rel_id] = 1000
                    elif id2relation[rel_id] in ["/location/country/capital", "/location/country/administrative_divisions", 
                                                 "/location/neighborhood/neighborhood_of", "/business/company/founders",
                                                 "/people/deceased_person/place_of_death", "/people/person/children", 
                                                 "/business/company/place_founded"]:
                        penal_vec[rel_id] = 300
                    else:
                        penal_vec[rel_id] = 100
                elif opt['data_name'] == 'NYTFB-68K':
                    if id2relation[rel_id] in ["/location/location/contains", "/people/person/place_lived", 
                                               "/people/person/nationality", "/people/person/children", 
                                               "/location/neighborhood/neighborhood_of", "/business/person/company"]:
                        penal_vec[rel_id] = 1000
                    elif id2relation[rel_id] in ["/location/country/capital", "/location/country/administrative_divisions", 
                                                 "/people/deceased_person/place_of_death", "/location/us_state/capital"]:
                        penal_vec[rel_id] = 200
                    else:
                        penal_vec[rel_id] = 500
                else:
                    raise Exception('Invalid dataset name')
               
            #  Incorporate entity frequencyinformation from the KB
            if penal_vec[rel_id] < 0:
                penal_vec[rel_id] *= 0.01 * (1 + min(entity_freq[e1_id], entity_freq[e2_id]))
                
            # Scale up the MIT/MID score
            penal_vec[rel_id] *= opt['penal_scalar']
            
        penal_vec_all[bag_name] = penal_vec
        
    return penal_vec_all
        
    
    
        
def getting_hamming_score(data, relation_data, opt):
    """
    Generate the hamming-score dictionary for global search
    """
    hamming_all = {}
    
    for bag_name in data.keys():
        hamming_score = np.ones(opt['num_rel']) * -1
        bag_label = relation_data[bag_name]
        hamming_score[bag_label==1] = 1
        hamming_all[bag_name] = hamming_score
        
    return hamming_all
    
    
    
    
    
    
        