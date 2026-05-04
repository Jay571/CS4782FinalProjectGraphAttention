# CS 4782 Final Project: Graph Attention Networks
## Purpose
The purpose of this Git repo is to reproduce the results scene in the Graph Attention Networks paper written by Petar Velickovic (https://arxiv.org/pdf/1710.10903). This paper introduces a Graph Attention Network (GAT) as a novel neural network architecture specifically designed for graph-based data. A GAT works by calculating different attention scores for the neighbors of each node. The main purpose of GATs is to outperform MLPs and CNNs (GCN) on several transductive and inductive learning tasks on graph-based data.

## Chosen Result
The results we attempted to reproduce were all the transductive and inductive learning tests for the paper's GAT model excluding the Const-GAT results. The difference between the GAT results that we attempted to reproduce and the Const-GAT results is that Const-GAT does not attempt to learn the different weights between the neighbors of each node.

## GitHub Contents
This repo is divided into 7 sections: data, models, utils, code, poster, results, and report. The data folder contains all raw and preprocessed data from the datasets used in the original paper. The models folder contains the PyTorch code for a single GAT attention head. The utils folder contains PyTorch code for the complete GAT model implementation. The code folder contains our actual reimplementation code of the project. The results folder contains images of the reimplementation results used in our poster and report. General project submission materials were not grouped into a specific folder. In the Juptyter Notebook found insid ethe Reimplementation folder, the original paper results can be found in code spaces 11 and 12, which are labeled "Plot learning curves" and "Final comparison table + bar chart" respectively, while our creative experiment results can be found in code spaces XXXX.

XXXX redo this according to directions after creative experiments are finished


## Reimplementation Details
We attempted to reproduce all of the GAT results presented by the original paper. We achieved this by creating a Jupyter Notebook so that we could run our reimplementation using Google Colab. We used a T4 GPU when running our reimplementation inside of Google Colab. The datasets used by the original paper are the Cora, Citeseer, PubMed, and PPI (protein-protein interaction) graph-based datasets. When evaluating our reimplementation, we used validation accuracy as our metric to evaluate our transductive results and micro-F1 scores to evaluate our inductive results. There were 3 key modifications we made to our implementation that differ from the original paper, all of which are related to saving memory. The 3 modifications we made are stricter early stopping, fewer maximum training epochs, and manual garbage collection. Without these 3 modifications, we wouldn't have been able to properly reproduce the results of the paper in Google Colab without encountering out-of-memory errors.

## Reproduction Steps
To reimplement this project for yourself, first, download this repo and move the files into a Google Drive folder. Open the Jupyter Notebook inside the reimplementation folder in Google Colab and change the project directory in the second code space labeled "User-editable Drive path" to the correct directory path to the project code in your Google Drive. Then run the entire Jupyter Notebook with a T4 GPU (or better). The original paper results can be found in code spaces 11 and 12, which are labeled "Plot learning curves" and "Final comparison table + bar chart" respectively. Afterwards, you can find our creative experiments results in code spaces XXXX.

## Results
Our reimplementation results are extremely close to the original paper's results. However, other than the results for the Citeseer dataset, our results still lie outside the lower confidence bounds of the original paper. Our current hypothesis for why our results fail to meet the confidence bound of the original paper is that our reimplementation uses a single predefined seed, while the original paper uses averages across multiple runs and multiple seeds. Our creative experiment results XXXXXXX.

## Conclusion
XXXXXX wait for creative experiments results

## References
Original paper: https://arxiv.org/pdf/1710.10903
XXXXXX

## Acknowledgements
This project was made for Cornell CS 4782 Deep Learning's final project during the Spring 2026 semester.
