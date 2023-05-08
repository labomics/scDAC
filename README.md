# scDAC
 single-cell deep adaptive clustering
 
# Requirements
Python --- 3.7.11

Pytorch --- 1.12.0

Sklearn --- 1.0.2

Numpy --- 1.21.6


# Run scDAC
Take the dataset chen_10 we provided here as an example.
After decompressing files vec.rar in the file path of scDAC/data/chen_10/subset_0 , run the following command:

CUDA_VISIBLE_DEVICES=0 py run.py --task chen_10 --exp e0

To infer the labels after training, run the following command:

CUDA_VISIBLE_DEVICES=0 py run.py --action infer_latent --task chen_10 --exp e0 --init-model sp_latest

# Outputs
You can obtain the predicted clustering labels under the folder /data/chen_10.
