# CS 4782 Final Project: Graph Attention Networks
## Purpose
The purpose of this Git repo is to reproduce the results scene in the Graph Attention Networks paper written by Petar Velickovic (https://arxiv.org/pdf/1710.10903). This paper introduces a Graph Attention Network (GAT) as a novel neural network architecture specifically designed for graph-based data. A GAT works by calculating different attention scores for the neighbors of each node. The main purpose of GATs is to outperform MLPs and CNNs (GCN) on several transductive and inductive learning tasks on graph-based data.

## Chosen Result
The results we attempted to reproduce were all the transductive and inductive learning tests for the paper's GAT model excluding the Const-GAT results. The difference between the GAT results that we attempted to reproduce and the Const-GAT results is that Const-GAT does not attempt to learn the different weights between the neighbors of each node.

## GitHub Contents
This repo is divided into 7 sections: data, models, utils, code, poster, results, and report. The data folder contains all raw and preprocessed data from the datasets used in the original paper. The models and utils folder contains PyTorch code to build a GAT model. The code folder contains our actual reimplementation code of the project. The poster folder contains a PDF of our poster submission. The results folder contains images of the reimplementation results used in our poster and report. The report folder contains a PDF of our report.

## Reimplementation Details
We attempted to reproduce all of the GAT results presented by the original paper. We achieved this by creating a Jupyter Notebook to run our reimplementation in Google Colab. We used a T4 GPU when running our reimplementation inside of Google Colab. The datasets used by the original paper are the Cora, Citeseer, PubMed, and PPI (protein-protein interaction) graph-based datasets. There were 3 key modifications we made to our implementation that differ from the original paper, all of which are related to saving memory. The 3 modifications we made are stricter early stopping, fewer maximum training epochs, and manual garbage collection. Without these 3 modifications, we wouldn't have been able to properly reproduce the results of the paper in Google Colab without encountering out-of-memory errors.

## Reproduction Steps
To reimplement this project for yourself, first, download this repo and move the files into a Google Drive folder. Open a Jupyter Notebook inside the code folder in Google Colab and change the project directory for the variable named `PROJECT_DIR` to the correct directory path of the project code in your Google Drive. Then run the entire Jupyter Notebook with a T4 GPU (or better). `experiment1_standard_gat.ipynb` is a reimplementation of the code used in our poster presentation while `experiment2_deeper_gat.ipynb` and `experiment3_bugfixed_gat.ipynb` are reimplementations of our creative experiment iterations.
In experiment 3, if you want to try the different stopping rules, use Command + F (Ctrl + F for Windows) to search for "stopping rule". You will find the two places to modify and you will find the instruction to modify the code accordingly. After replacing the two blocks of code with the code blocks in the code comment. Rerun the whole notebook and you will see the results.

## Results
Our reimplementation results are extremely close to the original paper's results. For `expirement1_standard_gat.ipynb`, other than the Citeseer dataset, our results still lie outside the lower confidence bounds of the original paper. Our current hypothesis for why our results fail to meet the confidence bound of the original paper is that our reimplementation uses a single predefined seed, while the original paper uses averages across multiple runs and multiple seeds. For the results of our final creative experiment in `experiment3_bugfixed_gat.ipynb`, we improved upon all of our initial experiment results presented in our poster. However, except for the Cora dataset, we failed to improve upon the results of the original paper. Our current hypothesis for why our results failed is that XXXXX

## Conclusion
XXXXXX wait for creative experiments results

## References
Original paper: https://arxiv.org/pdf/1710.10903
Datasets: XXXXXX


## Acknowledgements
This project was made for Cornell CS 4782 Deep Learning's final project during the Spring 2026 semester.
