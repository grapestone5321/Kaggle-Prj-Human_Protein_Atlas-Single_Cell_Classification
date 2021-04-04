# Kaggle-Human_Protein_Atlas-Single_Cell_Classification

-------

## Submission deadline

May 11, 2021 - Final submission deadline at 11:59 PM UTC

-------

## Task

This is a weakly supervised multi-label classification problem and a code competition. 

Given images of cells from our microscopes and labels of protein location assigned together for all cells in the image, Kagglers will develop models capable of segmenting and classifying each individual cell with precise labels. 

If successful, you'll contribute to the revolution of single-cell biology!

-------

## What am I predicting?
You are predicting protein organelle localization labels for each cell in the image. 

Border cells are included when there is enough information to decide on the labels.

There are in total 19 different labels present in the dataset (18 labels for specific locations, and label 18 for negative and unspecific signal). 

The dataset is acquired in a highly standardized way using one imaging modality (confocal microscopy). 

However, the dataset comprises 17 different cell types of highly different morphology, which affect the protein patterns of the different organelles. 

All image samples are represented by four filters (stored as individual files), the protein of interest (green) plus three cellular landmarks: 

- nucleus (blue), 

- microtubules (red), 

- endoplasmic reticulum (yellow). 

The green filter should hence be used to predict the label, and the other filters are used as references. 

The labels are represented as integers that map to the following:

      0. Nucleoplasm
      1. Nuclear membrane
      2. Nucleoli
      3. Nucleoli fibrillar center
      4. Nuclear speckles
      5. Nuclear bodies
      6. Endoplasmic reticulum
      7. Golgi apparatus
      8. Intermediate filaments
      9. Actin filaments 10. Microtubules
      11. Mitotic spindle
      12. Centrosome
      13. Plasma membrane
      14. Mitochondria
      15. Aggresome
      16. Cytosol
      17. Vesicles and punctate cytosolic patterns
      18. Negative

--------

## Cell Structure

### The Cell Atlas
https://www.proteinatlas.org/humanproteome/cell


## The Human Cell
https://www.youtube.com/watch?v=P4gz6DrZOOI&feature=emb_logo

Discover a world of cellular information. A movie that takes you through the human cell. 

Created by researchers, for researchers. Explore the human cell further at proteinatlas.org


------

## nature method

### Analysis of the Human Protein Atlas Image Classification competition
https://www.nature.com/articles/s41592-019-0658-6

-------

### Biology: Cell Structure I Nucleus Medical Media
https://www.youtube.com/watch?v=URUJD5NEXC8

This animation by Nucleus shows you the function of plant and animal cells for middle school and high school biology, including organelles like: 

      the nucleus, 
      nucleolus, 
      DNA (chromosomes), 
      ribosomes, 
      mitochondria, etc. 

Also included are: 

      ATP molecules, 
      cytoskeleton, 
      cytoplasm, 
      microtubules, 
      proteins, 
      chloroplasts, 
      chlorophyll, 
      cell walls, 
      cell membrane, 
      cilia, 
      flagellae, etc.

Watch another version of this video, narrated by biology teacher Joanne Jezequel here: https://youtu.be/cbiyKH9uPUw​


Watch other Nucleus Biology videos:

- Controlled Experiments: https://youtu.be/D3ZB2RTylR4​

- Independent vs. Dependent Variables: https://youtu.be/nqj0rJEf3Ew​

- Active Transport: https://youtu.be/ufCiGz75DAk



-------

## Evaluation

Submissions are evaluated by computing macro F1, with the mean taken over the 19 segmentable classes of the challenge. 

It is otherwise essentially identical to the OpenImages Instance Segmentation Challenge evaluation. 

The OpenImages version of the metric is described in detail here. See also this tutorial on running the evaluation in Python, with the only difference being the use of F1 rather than average precision.

Segmentation is calculated using IoU with a threshold of 0.6.

-------

## Submission File
For each image in the test set, you must predict a list of instance segmentation masks and their associated detection score (Confidence). 

The submission csv file uses the following format:

      ImageID,ImageWidth,ImageHeight,PredictionString
      ImageAID,ImageAWidth,ImageAHeight,LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA2 ...
      ImageBID,ImageBWidth,ImageBHeight,LabelB1 ConfidenceB1 EncodedMaskB1 LabelB2 ConfidenceB2 EncodedMaskB2 …

Note that a mask MAY have more than one class. 

If that is the case, predict separate detections for each class using the same mask.

      ImageID,ImageWidth,ImageHeight,PredictionString
      ImageAID,ImageAWidth,ImageAHeight,LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA1 ...

A sample with real values would be:

      ImageID,ImageWidth,ImageHeight,PredictionString
      721568e01a744247,1118,1600,0 0.637833 eNqLi8xJM7BOTjS08DT2NfI38DfyM/Q3NMAJgJJ+RkBs7JecF5tnAADw+Q9I
      7b018c5e3a20daba,1600,1066,16 0.85117 eNqLiYrLN7DNCjDMMIj0N/Iz9DcwBEIDfyN/QyA2AAsBRfxMPcKTA1MMADVADIo=

The binary segmentation masks are run-length encoded (RLE), zlib compressed, and base64 encoded to be used in text format as EncodedMask. 

Specifically, we use the Coco masks RLE encoding/decoding (see the encode method of COCO’s mask API), the zlib compression/decompression (RFC1950), and vanilla base64 encoding.

An example python function to encode an instance segmentation mask would be:

      import base64
      import numpy as np
      from pycocotools import _mask as coco_mask
      import typing as t
      import zlib


      def encode_binary_mask(mask: np.ndarray) -> t.Text:
        """Converts a binary mask into OID challenge encoding ascii text."""

        # check input mask --
        if mask.dtype != np.bool:
          raise ValueError(
              "encode_binary_mask expects a binary mask, received dtype == %s" %
              mask.dtype)

        mask = np.squeeze(mask)
        if len(mask.shape) != 2:
          raise ValueError(
              "encode_binary_mask expects a 2d mask, received shape == %s" %
              mask.shape)

        # convert input mask to expected COCO API input --
        mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask_to_encode = mask_to_encode.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)

        # RLE encode mask --
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

        # compress and base64 encoding --
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str

(This code is available as a gist here.)


-------

## Potentially useful Github links

On the official Cellprofiling Github we have some pieces of code that could, potentially, be useful for this competition.

- 1. The code for the solutions from the previous challenge that was later published in Nature Methods. You will also be able to get the trained weights from the Bestfitting's winning solution there.

- 2. Code for segmenting individual cells and nuclei in the HPA images.

### CellProfiling/HPA-competition-solutions
https://github.com/CellProfiling/HPA-competition-solutions

### CellProfiling/HPA-Cell-Segmentation
https://github.com/CellProfiling/HPA-Cell-Segmentation

### Nature Methods:
https://www.nature.com/articles/s41592-019-0658-6


-------

# Previous Human Protein Atlas Competition

https://www.kaggle.com/c/human-protein-atlas-image-classification/overview


## Previous Human Protein Atlas Competition Nature Methods Paper

### The results of the previous HPA competition were published in Nature Methods (Open Access):
### the the results of this competition will also likely be published in Nature Methods.

https://www.nature.com/articles/s41592-019-0658-6

## notebook

### Human Protein Resnet50 training / Pytorch
https://www.kaggle.com/vandalko/human-protein-resnet50-training-pytorch/notebook


### Fastai v1 starter pack (Kernel edition) [LB 0.323]
https://www.kaggle.com/hortonhearsafoo/fastai-v1-starter-pack-kernel-edition-lb-0-323

### GapNet-PL [LB 0.385]
https://www.kaggle.com/rejpalcz/gapnet-pl-lb-0-385

### HPA-MODEL-256
https://www.kaggle.com/omoekan09/hpa-model-256/notebook


-------

## Blogs

### Representation Learning
http://www.moseslab.csb.utoronto.ca/alexlu/project/representation-learning/

-------

## References


### 1. Ouyang, W. & Zimmer, C. The imaging tsunami: computational opportunities and challenges. Curr. Opin. Syst. Biol. 4, 105–113 (2017).



### 2. Uhlén, M. et al. Tissue-based map of the human proteome. Science 347, 1260419 (2015).



### 3. Thul, P. J. et al. A subcellular map of the human proteome. Science 356, eaal3321 (2017).


### 4. Mahdessian, D. et al. Spatiotemporal dissection of the cell cycle regulated human proteome. 

Preprint at bioRxiv https://doi.org/10.1101/543231 (2019).

### 5. Sullivan, D. P. et al. Deep learning is combined with massive-scale citizen science to improve large-scale image classification. Nat. Biotechnol. 36, 820–828 (2018).


 

### 6. Tsoumakas, G. & Katakis, I. Multi-label classification: an overview. Int. J. Data Warehous. Min. 3, 1–13 (2009).

 

### 7. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–444 (2015).

 

### 8. LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. Gradient-based learning applied to document recognition. IEEE, 86, 2278–2324 (1998).

 

### 9. Silver, D. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).

 

### 10. Bojarski, M. et al. End to end learning for self-driving cars. 

Preprint at https://arxiv.org/abs/1604.07316 (2016).

### 11. Simonyan, K. & Zisserman, A. Very deep convolutional networks for large-scale image recognition. 

Preprint at https://arxiv.org/abs/1409.1556 (2014).

### 12. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. & Wojna, Z. Rethinking the inception architecture for computer vision. in IEEE Conference on Computer Vision and Pattern Recognition 2818–2826 (IEEE, 2016).

### 13. Ronneberger, O., Fischer, P. & Brox, T. U-net: Convolutional networks for biomedical image segmentation. in Medical Image Computing and Computer-Assisted Intervention—MICCAI 2015 (eds Navab, N. et al.) 234–241 (Springer, 2015).

### 14. Hestness, J. et al. Deep learning scaling is predictable, empirically. 

Preprint at https://arxiv.org/abs/1712.00409 (2017).

## 15. Moen, E. et al. Deep learning for cellular image analysis. 

Nat. Methods https://doi.org/10.1038/s41592-019-0403-1 (2019).

### 16. Godinez, W. J., Hossain, I., Lazic, S. E., Davies, J. W. & Zhang, X. A multi-scale convolutional neural network for phenotyping high-content cellular images. Bioinforma. Oxf. Engl. 33, 2010–2019 (2017).

 

### 17. Hofmarcher, M., Rumetshofer, E., Clevert, D.-A., Hochreiter, S. & Klambauer, G. accurate prediction of biological assays with high-throughput microscopy images and convolutional networks. J. Chem. Inf. Model. 59, 1163–1171 (2019).


### 18. Kraus, O. Z., Ba, J. L. & Frey, B. J. Classifying and segmenting microscopy images with deep multiple instance learning. Bioinformatics 32, i52–i59 (2016).

### 19. He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. in IEEE Conference on Computer Vision and Pattern Recognition 770–778 (IEEE, 2016).

### 20. Huang, G., Liu, Z., Van Der Maaten, L. & Weinberger, K. Q. Densely connected convolutional networks. in IEEE Conference on Computer Vision and Pattern Recognition 4700–4708 (IEEE, 2017).

### 21. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. J. Mach. Learn. Res. 15, 1929–1958 (2014).

 

## 22. Ioffe, S. & Szegedy, C. Batch normalization: accelerating deep network training by reducing internal covariate shift. 

Preprint at https://arxiv.org/abs/1502.03167 (2015).

### 23. Lin, T.Y., Goyal, P., Girshick, R., He, K. & Dollár, P. Focal loss for dense object detection. in IEEE International Conference on Computer Vision 2980–2988 (IEEE, 2017).

### 24. Smith, L. N. Cyclical learning rates for training neural networks. in IEEE Winter Conference on Applications of Computer Vision 464–472 (IEEE, 2017).

### 25. Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V. & Le, Q. V. AutoAugment: learning augmentation policies from data. 

Preprint at https://arxiv.org/abs/1805.09501 (2018).

### 26. Paszke, A. et al. Automatic differentiation in PyTorch. in NIPS 2017 Autodiff Workshop (2017).

### 27. Abadi, M. et al. TensorFlow: large-scale machine learning on heterogeneous distributed systems. 

Preprint at https://arxiv.org/abs/1603.04467 (2016).

### 28. Hutter, F., Kotthoff, L. Vanschoren, J. Automated Machine Learning-Methods, Systems, Challenges (Springer International Publishing, 2019).

### 29. Falkner, S., Klein, A. & Hutter, F. BOHB: robust and efficient hyperparameter optimization at scale. in 35th International Conference on Machine Learning 1436–1445 (ICML, 2018).

### 30. Vanschoren, J. Meta-learning: a survey. 

Preprint at https://arxiv.org/abs/1810.03548 (2018).

### 31. Elsken, T., Metzen, J. H. & Hutter, F. Neural architecture search: a survey. J. Mach. Learn. Res. 20, 1–21 (2019).


### 32. Russakovsky, O. et al. ImageNet large scale visual recognition challenge. Int. J. Comput. Vis. 115, 211–252 (2015).


### 33. Deng, J. et al. ImageNet: a large-scale hierarchical image database. in IEEE Conference on Computer Vision and Pattern Recognition 248–255 (IEEE, 2009).

### 34. Foggia, P., Percannella, G., Soda, P. & Vento, M. Benchmarking HEp-2 cells classification methods. IEEE Trans. Med. Imaging 32, 1878–1889 (2013).

 

### 35. Ulman, V. et al. An objective comparison of cell-tracking algorithms. Nat. Methods 14, 1141–1152 (2017).


 

### 36. Johnson, J. M. & Khoshgoftaar, T. M. Survey on deep learning with class imbalance. J. Big Data 6, 27 (2019).
 

### 37. Sechidis, K., Tsoumakas, G. & Vlahavas, I. On the stratification of multi-label data. in Machine Learning and Knowledge Discovery in Databases Vol. 6913 (eds Gunopulos, D. et al.) 145–158 (Springer International Publishing, 2011).

### 38. Berman, M., Rannen Triki, A. & Blaschko, M. B. The Lovász-Softmax loss: a tractable surrogate for the optimization of the intersection-over-union measure in neural networks. in IEEE Conference on Computer Vision and Pattern Recognition 4413–4421 (IEEE, 2018).

### 39. Yosinski, J., Clune, J., Bengio, Y. & Lipson, H. How transferable are features in deep neural networks? in Advances in Neural Information Processing Systems Vol. 27 (eds Ghahramani, Z. et al.) 3320–3328 (Curran Associates, Inc., 2014).

### 40. Selvaraju, R. R. et al. Grad-cam: visual explanations from deep networks via gradient-based localization. in IEEE International Conference on Computer Vision 618–626 (IEEE, 2017).

### 41. Deng, J., Guo, J., Xue, N. & Zafeiriou, S., Arcface: additive angular margin loss for deep face recognition. in IEEE Conference on Computer Vision and Pattern Recognition 4690–4699 (IEEE, 2019).

### 42. McInnes, L., Healy, J. & Melville, J. UMAP: uniform manifold approximation and projection for dimension reduction.

Preprint at https://arxiv.org/abs/1802.03426 (2018).

### 43. Ouyang, W., Mueller, F., Hjelmare, M., Lundberg, E. & Zimmer, C. ImJoy: an open-source computational platform for the deep learning era. 

https://doi.org/10.1038/s41592-019-0627-0 (2019).

### 44. Belthangady, C. & Royer, L. A. Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction. Nat. Methods 

https://doi.org/10.1038/s41592-019-0458-z (2019).

### 45. Zech, J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med. 15, e1002683 (2018).

 

### 46. Riley, P. Three pitfalls to avoid in machine learning. Nature 572, 27–29 (2019).


### 47. Oei, R. W. et al. Convolutional neural network for cell classification using microscope images of intracellular actin networks. PLoS ONE 14, e0213626 (2019).
 

### 48. Kornblith, S., Shlens, J. & Le, Q. V. Do better imagenet models transfer better? in IEEE Conference on Computer Vision and Pattern Recognition 2661–2671 (IEEE, 2019).

### 49. Stadler, C., Skogs, M., Brismar, H., Uhlén, M. & Lundberg, E. A single fixation protocol for proteome-wide immunofluorescence localization studies. J. Proteom. 73, 1067–1078 (2010).
 

### 50. Van Der Walt, S. et al. scikit-image: image processing in Python. PeerJ 2, e453 (2014).


---------

## Papers

### Bojarski, M. et al. End to end learning for self-driving cars. 

Preprint at https://arxiv.org/abs/1604.07316 (2016).

### Simonyan, K. & Zisserman, A. Very deep convolutional networks for large-scale image recognition. 

Preprint at https://arxiv.org/abs/1409.1556 (2014).



### Hestness, J. et al. Deep learning scaling is predictable, empirically. 

Preprint at https://arxiv.org/abs/1712.00409 (2017).


### Ioffe, S. & Szegedy, C. Batch normalization: accelerating deep network training by reducing internal covariate shift. 

Preprint at https://arxiv.org/abs/1502.03167 (2015).



### Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V. & Le, Q. V. AutoAugment: learning augmentation policies from data. 

Preprint at https://arxiv.org/abs/1805.09501 (2018).

### Abadi, M. et al. TensorFlow: large-scale machine learning on heterogeneous distributed systems. 

Preprint at https://arxiv.org/abs/1603.04467 (2016).


### Vanschoren, J. Meta-learning: a survey. 

Preprint at https://arxiv.org/abs/1810.03548 (2018).


### McInnes, L., Healy, J. & Melville, J. UMAP: uniform manifold approximation and projection for dimension reduction.

Preprint at https://arxiv.org/abs/1802.03426 (2018).

-------


## Website

### Mahdessian, D. et al. Spatiotemporal dissection of the cell cycle regulated human proteome. 

Preprint at bioRxiv https://doi.org/10.1101/543231 (2019).


## Moen, E. et al. Deep learning for cellular image analysis. 

Nat. Methods https://doi.org/10.1038/s41592-019-0403-1 (2019).


### Ouyang, W., Mueller, F., Hjelmare, M., Lundberg, E. & Zimmer, C. ImJoy: an open-source computational platform for the deep learning era. 

https://doi.org/10.1038/s41592-019-0627-0 (2019).

## Belthangady, C. & Royer, L. A. Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction. Nat. Methods 

https://doi.org/10.1038/s41592-019-0458-z (2019).

-------

## Paper-2

### HUMAN-LEVEL PROTEIN LOCALIZATION WITH CONVOLUTIONAL NEURAL NETWORKS
https://openreview.net/pdf?id=ryl5khRcKm


### END-TO-END LEARNING OF PHARMACOLOGICAL ASSAYS FROM HIGH-RESOLUTION MICROSCOPY IMAGES
https://openreview.net/pdf?id=S1gBgnR9Y7


### DEEP-LEARNING BASED PHENOTYPE CLASSIFICATION in High Content cellular imaging ON INTEL® ARCHITECTURE
https://simplecore.intel.com/nervana/wp-content/uploads/sites/53/2018/06/IntelAIDC18_Datta_Theatre_052418_final.pdf

-------

 ## Dataset

- HPA 2021 TFRecords 512 Cell 0: Elahi updated a day ago (Version 1)




-------

## Progress


### Public Best LB Score: 0.362

### Private Score: 



-------

## fastai cell tile prototyping [training]
https://www.kaggle.com/dragonzhang/fastai-cell-tile-prototyping-training

      Public Score 0.362

-------


