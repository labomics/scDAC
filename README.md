scDAC

 scDAC is a single-cell Deep Adaptive Clustering (scDAC) model by coupling the Autoencoder and the Dirichlet Process Mixture Model.
 
 ![image](https://github.com/omicshub/scDAC/blob/main/scDAC/image/fig1.png)

# Requirements

Python --- 3.7.11

Pytorch --- 1.12.0

Sklearn --- 1.0.2

Numpy --- 1.21.6

# Run scDAC

Take the dataset chen_10 we provided here as an example.

Decompress the expression matrix file in scDAC/data/mat and split it for the input by running the scDAC/preprocess/preprocess_split.ipynb and obtain the files of cell names and feature names by running scDAC/preprocess/preprocess_cell_and_feature_names.ipynb

After  running the preprocess_split or decompressing files vec.rar in the file path of scDAC/data/chen_10/subset_0 , run the following command to train the scDAC model:

    $ CUDA_VISIBLE_DEVICES=0 py run.py --task chen_10 --exp e0

To infer the labels after training, run the following command:

    $ CUDA_VISIBLE_DEVICES=0 py run.py --action infer_latent --task chen_10 --exp e0


You can obtain the predicted clustering labels under the folder /data/chen_10, and the ARI, NMI and SC metrics. The indicator of  Deviation Ratio (DR) can be calculated by running "scDAC/metrics_DR.ipynb".

# results reproduction

The results of scDAC in the paper can be reproduced by using the code in the file "scDAC/demo". The code of compared methods can be found in the file "scDAC/compared methods".

The datasets used in the paper can be found in *** , we can download and save them to the path "/data/preprocessed". When using the datasets orozco and kozareva, we can use "/preprocess/preprocess_split.ipynb" to obtain the input as the input of these two dataset are too large to upload. We can also download the original data and preprocess it as the following preprocess steps suggested in the following. The accession IDs are provided in the paper.

# using for new dataset

If you want to use scDAC to analyse other data, you can running the code as the following:

1.Please preprocess the data by Seurat referring to https://satijalab.org/seurat/articles/pbmc3k_tutorial, or use the code "preprocess.ipynb" provided by scDAC in the "preprocess" folder.

Whether we do the preprocessing step ourselves or with the scDAC's code in the "preprocess" folder, we need to get the following files:

a. The expression matrices of highly variable genes.

b.The feature number and feature name of the highly variable genes and the cellnames of every expression matrix.

We can use "preprocess_split.ipynb" in the "preprocess" folder to split the matrix to vectors. The vectors are the inneed input of scDAC.

2.Please supplement the file "scDAC/configs/data.toml" with information about the new dataset.

3.Please train the model:

     $ CUDA_VISIBLE_DEVICES=0 py run.py --task [datasetname]

4.To infer the labels after training, run the following command:

    $  CUDA_VISIBLE_DEVICES=0 py run.py --action infer_latent --task [datasetname] --init-model sp_latest

Then we can obtain the predict label and ARI, NMI and SC sore.

5.The indicator of  Deviation Ratio (DR) can be calculated by running scDAC/metrics_DR.ipynb


