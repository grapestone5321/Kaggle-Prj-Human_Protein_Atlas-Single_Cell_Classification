# Kaggle-Human_Protein_Atlas-Single_Cell_Classification

-------

## Submission deadline

April 27, 2021 - Final submission deadline at 11:59 PM UTC

-------

## Evaluation

Submissions are evaluated by computing macro F1, with the mean taken over the 19 segmentable classes of the challenge. 

It is otherwise essentially identical to the OpenImages Instance Segmentation Challenge evaluation. 

The OpenImages version of the metric is described in detail here. See also this tutorial on running the evaluation in Python, with the only difference being the use of F1 rather than average precision.

Segmentation is calculated using IoU with a threshold of 0.6.

-------


