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

## Previous Human Protein Atlas Competition

https://www.kaggle.com/c/human-protein-atlas-image-classification/overview




### Human Protein Resnet50 training / Pytorch
https://www.kaggle.com/vandalko/human-protein-resnet50-training-pytorch/notebook


### Fastai v1 starter pack (Kernel edition) [LB 0.323]
https://www.kaggle.com/hortonhearsafoo/fastai-v1-starter-pack-kernel-edition-lb-0-323


## Previous Human Protein Atlas Competition Nature Methods Paper

The results of the previous HPA competition were published in Nature Methods (Open Access):

https://www.nature.com/articles/s41592-019-0658-6

the the results of this competition will also likely be published in Nature Methods.

---------
