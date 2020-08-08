SEAL -- learning from Subgraphs, Embeddings, and Attributes for Link prediction
===============================================================================

About
-----

Python version of SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction).


Installation
------------

Install [PyTorch](https://pytorch.org/).

Type

    bash ./install.sh

to install the required software and libraries. It will download and install the default graph neural network software [\[pytorch_DGCNN\]](https://github.com/muhanzhang/pytorch_DGCNN) to the same level as the root SEAL folder (not this Python folder).


Usages
------

Type "python Main.py --data-name USAir" to have a try of SEAL on the USAir network.

Type:
    
    python Main.py --data-name XXX --hop 'auto' --batch-size 1

to reproduce the experimental results in Table 1 of the paper on dataset XXX. The default seed is 1. Append "--seed Y" to the above command to use other seeds.

Type:

    python Main.py --data-name NS --test-ratio 0.5 --hop 'auto' --use-embedding

to run SEAL on the NS network with 50% observed links randomly removed as testing links, hop number set automatically from {1, 2}, and node2vec embeddings included.

Type:

    python Main.py --data-name PPI_subgraph --test-ratio 0.5 --use-embedding --use-attribute

to run SEAL on PPI_subgraph with node attributes included. The node attributes are assumed to be saved in the  _group_ of the _.mat_ file.

Type:

    python Main.py --train-name PB_train.txt --test-name PB_test.txt --hop 1

to run SEAL on a custom splitting of train and test links, where each row of "PB_train.txt" is an observed training link, each row of "PB_test.txt" is an unobserved testing link. Note that links in "PB_train.txt" will be used to construct the observed network, yet it is not necessary to train SEAL on all links in "PB_train.txt" especially when the number of observed links is huge. To set a maximum number of links to train on, append "--max-train-num 10000" for example.

Sometimes even extracting 1-hop enclosing subgraphs for some links leads to unaffordable number of nodes in the enclosing subgraphs, especially in Twitter-type networks where a hub node can have millions of followers. To deal with this case, append "--max-nodes-per-hop 100" for example to restrict the number of nodes in each hop to be less than 100 using random sampling. SEAL still shows excellent performance.

Type:

    python Main.py --data-name PB --train-name PB_train.txt --test-name PB_test.txt --hop 1

to still use the positive train and test links in "PB_train.txt" and "PB_test.txt", but use the network contained in "PB.mat" (here "--use-attribute" can be used if there is a _group_ within "PB.mat").

If you want to use SEAL to output link existence probabilities for unknown links, first type

    python Main.py --train-name PB_train.txt --test-name PB_val.txt --hop 1 --save-model

where "PB_val.txt" is asssumed to be the held-out observed links to validating your model, and "--save-model" saves the final model to "data/PB_model.pth". Now, assume "PB_test.txt" is the pairs of nodes that you want to output link existence probabilities for (i.e., it may contain both positive and negative links). Type:
    
    python Main.py --train-name PB_train.txt --test-name PB_test.txt --hop 1 --only-predict

to load the saved model and output predictions for "PB_test.txt" to "data/PB_test_pred.txt".

Requirements
------------

Latest tested combination: Python 3.8.1 + Pytorch 1.4.0

Required python libraries: gensim and scipy; all python libraries required by pytorch_DGCNN such as networkx, tqdm, sklearn etc.

If you want to enable embeddings for link prediction, please install the network embedding software 'node2vec' in "software/" (if the included one does not work).


Reference
---------

If you find the code useful, please cite our paper:

    @article{zhang2018link,
      title={Link Prediction Based on Graph Neural Networks},
      author={Zhang, Muhan and Chen, Yixin},
      journal={arXiv preprint arXiv:1802.09691},
      year={2018}
    }

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
9/5/2018
