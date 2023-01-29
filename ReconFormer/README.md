# ReconFormer
ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer

# Requirements

python=3.6  
pytorch=1.7.0

Please refer conda_environment.yml for more dependencies.


Preprocessed fastMRI (OneDrive) - <a href="https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/EtXsMeyrJB1Pn-JOjM_UqhUBdY1KPrvs-PwF2fW7gERKIA?e=uuBINy"> Link </a>\
Password: pguo4@jhu.edu\
**Note:** In preprocessed fastMRI, We didn't modify the original fastMRI data and just make the format compatible with our DataLoader. 

# Run

## Clone this repo
```bash 
git clone git@github.com:guopengf/ReconFormer.git
```

## Set up conda environment
```bash
cd ReconFormer
conda env create -f conda_environment.yml
conda activate recon
```
## Train ReconFormer
```bash 
bash run_recon_exp.sh
```

## Monitor the traning process
```bash 
tensorboard --logdir 'Dir path for saving checkpoints'
```
## Test 
(Download [pre-trained weights](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/Er37oIyNy3NBrXbeCQBp_fQBAxELR8UDaq6gHd-fjwRrSw) Password: pguo4@jhu.edu)
```bash 
bash run_recon_eval.sh
```


