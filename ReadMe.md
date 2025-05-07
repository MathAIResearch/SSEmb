# SSEmb: A Structural and Semantic Embedding Model for Mathematical Formula Retrieval.

## Introduction
Formula retrieval is an important topic in `Mathematical Information Retrieval (MIR)`. We propose `SSEmb`, a novel embedding-based model that combines structural and semantic information of formulas. Structurally, We use Graph Contrastive Learning (GCL) to encode formulas represented as Operator Graphs (OPGs), and introduce a graph data augmentation named substructure substitution to preserve mathematical validity while enhancing structural diversity. Semantically, we use Sentence-BERT to encode surrounding context of formulas. The structural and semantic embeddings are scored independently via cosine similarity and fused through a weighted scheme. In the `ARQMath-3 formula retrieval task`, SSEmb outperforms existing embedding methods by over 5 percentage points on P'@10 and nDCG'@10, approaching the performance of the matching-based method Approach0. Moreover, SSEmb boosts the effectiveness of all runs of other methods and achieves the state-of-the-art when combined with Approach0.

## Setup

### Environment
Create conda environment and install the required packages by running the following command:
```
$ conda env create -f environment.yml
```
### Raw data
Raw data can be downloaded from the [dprl](https://www.cs.rit.edu/~dprl/ARQMath).
### Evaluaion tool
Evaluation is performed using [trec_eval](https://github.com/usnistgov/trec_eval). Install the tool in the "SSEmb/Evaluation/" directory.

## File Description and Running
The following files are located under the "`SSEmb/StructEmb/`" directory:
* Run the following commands to generate the OPG Representation of train data and query data. 
    ```
    $ python train_data_generation.py
    $ python query_data_generation.py
    ```
* Run the following command to train the model.
    ```
    $ bash train.sh
    ```  
* Run the following command to get structural embeddings and retrieval the first-stage results.
    ```
    $ bash retrieval.sh
    ```  
The following files are located under the "`SSEmb/SemEmb/`" directory:
* Run the following command to get semantic embeddings.
    ```
    $ python get_semantic_embedding.py
    ```  
The following files are located under the "`SSEmb/Rank and Retrieval/`" directory:
* Run the following command to retrieval the second-stage results.
    ```
    $ python reranking.py
    ```  
The following files are located under the "`SSEmb/Evaluation/`" directory: 
* Run the following command to combine runs of different systems:
    ```
    $ python RRF/combining_runs.py
    ```  
* Put runs in the "ARQMath_2022_Submission/" directory and run the following command to evaluate results:
    ```
    $ bash evaluation.sh
    ``` 