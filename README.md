scDAC

 scDAC is a single-cell Deep Adaptive Clustering (scDAC) model by coupling the Autoencoder (AE) and the Dirichlet Process Mixture Model (DPMM)

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

# Outputs

You can obtain the predicted clustering labels under the folder /data/chen_10, and the ARI, NMI and SC metrics. The indicator of  Deviation Ratio (DR) can be calculated by running scDAC/metrics_DR.ipynb
