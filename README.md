# CS 4782 Final Project: Graph Attention Networks
## Purpose
The purpose of this Git repo is to reproduce the results scene in the Graph Attention Networks paper written by Petar Velickovic (https://arxiv.org/pdf/1710.10903). This paper introduces a Graph Attention Network (GAT) as a novel neural network architecture specifically designed for graph-based data. A GAT works by calculating different attention scores for the neighbors of each node. The main purpose of GATs is to outperform MLPs and CNNs (GCN) on several transductive and inductive learning tasks on graph-based data.

## Chosen Result
The results we attempted to reproduce were all the transductive and inductive learning tests for the paper's GAT model excluding the Const-GAT results. The difference between the GAT results that we attempted to reproduce and the Const-GAT results is that Const-GAT does not attempt to learn the different weights between the neighbors of each node.

## GitHub Contents
This repo is divided into 5 sections: data, models, utils, and general project submission materials. The data folder contains all raw and preprocessed data from the datasets used in the original paper. The models folder contains the original GAT code from the paper that we then converted into PyTorch code. The reimplementation folder contains our actual reimplementation code of the project and images used in our poster and report.
General project submission materials were not grouped into a specific folder.

## Reimplementation Details
We attempted to reproduce all of the GAT results presented by the original paper. We achieved this by creating a Jupyter Notebook so that we could run our reimplementation using Google Colab. The datasets used by the original paper are the Cora, Citeseer, PubMed, and PPI (protein-protein interaction) graph-based datasets. 
