# Multi-input Vision Transformer with Similarity Matching
## Official Pytorch Implementation

#### This is a source code for our paper [Multi-input Vision Transformer with Similarity Matching](https://link.springer.com/chapter/10.1007/978-3-031-46005-0_16) by Seungeun Lee, Sung Ho Hwang, Saelin Oh, Beom Jin Park, and Yongwon Cho.

#### Abstract
Multi-input models for image classification have recently gained considerable attention. However, multi-input models do not always exhibit superior performance compared to single models. In this paper, we propose a multi-input vision transformer (ViT) with similarity matching, which uses original and cropped images based on the region of interest (ROI) as inputs, without additional encoder architectures. Specifically, two types of images are matched on the basis of their cosine similarity in descending order, and they serve as inputs for a multi-input model with two parallel ViT-architectures. We conduct two experiments using a dataset of pediatric orbital wall fracture and chest X-rays. Consequently, the multi-input models with similarity matching outperform the baseline models and achieve balanced results. Furthermore, it is feasible that our method provides both global and local features, and the Grad-CAM results demonstrate that two different inputs of the proposed mechanism can help complementarily study the image. The code is available at https://github.com/duneag2/vit-similarity.
