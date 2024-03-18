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

# Results reproduction

The results of scDAC in the paper can be reproduced by using the code in the file "scDAC/demo". The code of compared methods can be found in the file "scDAC/compared methods".

The datasets used in the paper can be found at (https://drive.google.com/file/d/1qyaxzW7k0rA9Y08P_ZlapVb43l0xZuoE/view?usp=drive_link), we can download and unzip them to the path "/data/preprocessed". When using the datasets orozco and kozareva, we can use "/preprocess/preprocess_split.ipynb" to obtain the input as the input of these two dataset are too large to upload. We can also download the original data and preprocess it as the following preprocess steps suggested in the following. The accession IDs are provided in the paper.

# Using for new dataset

If you want to use scDAC to analyse other data, you can running the code as the following:

1.Please preprocess the data by Seurat referring to https://satijalab.org/seurat/articles/pbmc3k_tutorial, or use the code "preprocess.ipynb" provided by scDAC in the "preprocess" folder.

Whether we do the preprocessing step ourselves or with the scDAC's code in the "preprocess" folder. The specific steps and parameter settings are as follows: UMI count matrices are normalized and log-transformed using the NormalizeData function in Seurat package (v.4.1.0). We obtain the gene expression matrix after selecting 4000 highly variable genes using the FindVariableFeatures function and scaling the selected genes using the ScaleData function. The other parameters are set to the default value suggested in Seurat. We need to get the following files:

a. The expression matrices of highly variable genes.

b.The feature number and feature name of the highly variable genes and the cellnames of every expression matrix.

We can use "preprocess_split.ipynb" in the "preprocess" folder to split the matrix to vectors. The vectors are the inneed input of scDAC.

2.Please supplement the file "scDAC/configs/data.toml" with information about the new dataset.

3.Please train the model:

     $ CUDA_VISIBLE_DEVICES=0 py run.py --task [datasetname]

The parameters in scDAC include the network parameters in the AE module and the model parameters in the DPMM module. The parameter details of scDAC are as follows: in the AE module, we set the sizes of two hidden layers of encoder to 256 and 128, set the sizes of the hidden layers of decoder to 128 and 256, and set the dimensionality of the low-dimensional representation to 32; in the DPMM module, the number of components α is robust and often set to 1e-10, the other parameters are set to the default value suggested in the BayesianGaussianMixture function; in the loss function, we set the weight hyperparameter λ to 1 in all experiments. During the training process, the minibatch size is set to 512, and the AdamW optimizer with a fixed learning rate of 1e − 4 is used for optimization. 

4.To infer the labels after training, run the following command:

    $  CUDA_VISIBLE_DEVICES=0 py run.py --action infer_latent --task [datasetname] --init-model sp_latest

Then we can obtain the predict label and ARI, NMI and SC sore.

5.The indicator of  Deviation Ratio (DR) can be calculated by running scDAC/metrics_DR.ipynb


