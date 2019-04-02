Structured Minimally Supervised Learning for Neural Relation Extraction
=========================



This repo contains the *pytorch* code for paper [Structured Minimally Supervised Learning for Neural Relation Extraction](https://arxiv.org/abs/1904.00118).

        @inproceedings{bai2019structured,
          author     = {Bai, Fan and Ritter, Alan},
          title      = {Structured Minimally Supervised Learning for Neural Relation Extraction},
          booktitle  = {Proceedings of NAACL-HLT 2019},
          year       = {2019}
        } 



## Requirements


- Python 2 (tested on 2.7)
- PyTorch (tested on 0.4.1)
- cython (tested on 0.25.2)
- Screen (tested on 4.03.01, for reproducing)


## Dataset
Two datasets **NYTFB-68K/NYTFB-280K** can be found [here](https://drive.google.com/file/d/1FNRHVZP4aqhLwdmIcM8uiBVXTizNMNQP/view?usp=sharing). 

**NYTFB-68K**: Riedel et. al. HeldOut dataset.

**NYTFB-280K**: Lin et. al. dataset removing overlapping entity pairs from training data(only entity names are shared not sentences).

**Sentential DEV/TEST data**: Manually annotated data created by Hoffmann et. al.

Please checkout the Appendix B of our paper for detailed introduction and comparison about these two datasets. 

+ train.txt: e1_id, e2_id, e1_name, e2_name, relation, sentence
+ test.txt: same as train.txt
+ relation2id.txt: relation, relation_id
+ sentential_DEV.txt: e1_id, e2_id, sen_index_in_bag, relation, manual_label, sentence_entity_annoated, e1, e2, sentence
+ sentential_TEST.txt: same as dev file
+ vec.bin: pre-trained embedding file


## Training

Train a PCNN-NMAR model with a specific configuration:
```
python train.py --data_dir data/NYTFB-68K --lr 0.001 --penal_scalar 1000 --num_epoch 15 --save_dir saved_models/
```

With the above command, the model's checkpoint with best sentential AUC performance will be saved to `./saved_models/` as `NYTFB-68K_lr0.001_penal1000_best_model.tar`. You can save the checkpoint of every epoch by setting `--save_each_epoch True`, and perform heldou evaluation with `--heldout_eval True`.



## Evaluation

All checkpoints used in our paper are stored under `./checkpoints_in_paper/`

Sentential evaluation.
```
python eval.py --data_dir data/NYTFB-68K --model_dir checkpoints_in_paper/ --model_name NYTFB-68K_sentential.tar --sentential_eval True --sen_file sentential_DEV.txt 
```

Heldout evaluation.
```
python eval.py --data_dir data/NYTFB-68K --model_dir checkpoints_in_paper/ --model_name NYTFB-68K_heldout.tar --heldout_eval True
```
You can also print out the configuration of the model by setting `--print_config True`

## Reproduce

Since PCNN-NMAR is sensitive to the initialization, if you want to train the model from scratch to reproduce the result in our paper, you can run the script `tune.sh` with data directory and available GPU ids:
```
sh tune.sh "data/NYTFB-68K" "0 1 2 3"
```

This script will tune hyperparameters KB disagreement penalty scalars among {100, 200, ..., 2000} and learning rates among {0.001, 0.01}.

To select the model with best sentential result on DEV set among all saved models:
```
python eval.py --data_dir data/NYTFB-68K --model_dir saved_models/ --sentential_eval True --sen_file sentential_DEV.txt --tune True
```


## Reference
[Riedel et al. 2010] Sebastian Riedel and Limin Yao and Andrew McCallum. Modeling Relations and Their Mentions without Labeled Text. In Proceedings of ECML.

[Hoffmann et al., 2011] Hoffmann, Raphael  and  Zhang, Congle  and  Ling, Xiao  and  Zettlemoyer, Luke  and  Weld, Daniel S. Knowledge-Based Weak Supervision for Information Extraction of Overlapping Relations. In Proceedings of ACL.

[Lin et al., 2016] Lin, Yankai  and  Shen, Shiqi  and  Liu, Zhiyuan  and  Luan, Huanbo  and  Sun, Maosong. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL. [C++ code/data](https://github.com/thunlp/NRE)

[Bai et al. 2019] Bai, Fan and Ritter, Alan. Structured Minimally Supervised Learning for Neural Relation Extraction. In Proceedings of NAACL-HLT.


