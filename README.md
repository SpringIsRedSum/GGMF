# GGMF
The implementation of GGMF
# GGMF

![image-20250513141819825](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250513141819825.png)

## Abstract

Cancer ranks among the leading causes of disease-related human mortality worldwide. Accurate prediction of cancer patients’survival periods holds significant importance for formulating personalized treatment strategies. In recent years, rapid advancements in medical imaging have enabled whole-slide images (WSIs) and genomic data to provide abundant information for cancer survival analysis. However, effectively fusing multi-modal data to extract complementary insights remains a challenging task in existing research. To alleviate this problem, we propose a genome-guided multi-modal feature fusion (GGMF) approach for survival analysis.By leveraging a self-attention mechanism to identify key genes, GGMF emphasizes the genome’s dominant role in prediction.This process enhances the feature representation of WSIs by guiding the secondary encoding of the images with relevant genomic information. Additionally, a residual-based cross-modal interaction module is introduced to preserve both fused feature information and unimodal-specific information, improving model effectiveness. Experimental results demonstrate that GGMF outperforms multiple baseline methods on real-world lung adenocarcinoma and breast cancer datasets, validating its effectiveness and rationality in cancer survival prediction.

## Installation Guide for Linux

Linux amax-pc 4.15.0-142-generic

NVIDIA GPU: Quadro RTX 8000 with CUDA 11.6

It is recommended to use conda to install the required dependencies in the file

## Data processing

download_svs.sh is recommended for downloading WSIs files. Just download the corresponding manifest from the TCGA portal and set the corresponding save path when running the script. Relevant clinical and genomic information can also be downloaded together from TCGA's web portal. In the pre-processing stage, CLAM toolkit was used to cluster and generate the corresponding pathological features, and these features were saved in the form of.pt files. Genomic data are read as a.csv file using pytorch's data loader to align the dimensions.

We provide a set of processed genetic data and several sample images (all uploads would take a lot of space and time), tags.

## Train

Random partitioning of the data in cross validation is required. shuffle=True is set to random loading in dataset_survival.py's data loader by default. You can also change the parameters if you want to set up other forms of cross validation.

After setting the data path of related pictures, genes, and clinical information in the file train.py, and other parameters are saved, you can directly use "python train.py" to train. The training process will be printed, and the final results will be saved to the result folder or the path you specified.
