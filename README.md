# temporal-gcn-lstm
Code for Characterizing and Forecasting User Engagement with In-App Action Graphs: A Case Study of Snapchat

**Temporal-gcn-lstm** model encodes temporal evolving action graphs to predict future user engagement. 
The *end-to-end, multi-channel neural model* also encodes acitivity sequences and other macroscopic features to reach best performance.

### Requirements

DGL, NetworkX, PyTorch, Pandas, Numpy, SciKit-Learn, tqdm

Deep Graph Library (DGL) https://www.dgl.ai/

Pytorch https://pytorch.org/


### Building action graphs

build_graphs.py:    build static graphs for time period

build_temporal.py:  build temporal graphs per day

```python3 build_graphs.py INPUT_PATH OUTPUT_PATH```

```python3 build_temporal.py INPUT_PATH OUTPUT_PATH```

### Models

utils.py: supporting functions

activity_seq_model.py:  baseline activity sequence model

gcn_model.py: model structure of our graph convolutional network

multi_channel.py: To run our final best performance temporal graph model

### To Run

```python3 multi_channel.py```

Load custom data with ```df_path``` ```graphs_path``` ```macro_path``` flags

Set variants of model with ```--activity``` ```--macro``` flags to inlcude or leave out these information. 
ex. ```--activity False```. Default for both are ```True``` for best enhanced performance of model.

Hyperparameters were set to optimal for our dataset, they can be modified as input arguments.

## Cite

```
@inproceedings{liu2019characterizing,
  title={Characterizing and forecasting user engagement with in-app action graph: A case study of snapchat},
  author={Liu, Yozen and Shi, Xiaolin and Pierce, Lucas and Ren, Xiang},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2023--2031},
  year={2019}
}
```
