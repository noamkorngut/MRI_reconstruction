# MRI_reconstruction



## Dataset:
Prepare the dataset in the following structure for easy use of the code.The provided data loaders is ready for this this format.


```bash


                  
                  |                       
                  |                |--xxx.h5  
                  |--train--|...
                  |                |...
 Dataset-----|    |                   
                  |
                  |                |--xxx.h5      
                  |--val--  |...
                  |                |...  
                              
                          
                                    
 ```

## for downloading train, val and test sets, run the following commands:
```bash 
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_train.tar.xzAWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qV5z%2Bt5EVqoExVsS%2F%2B%2Fb6O4Tneg%3D&Expires=1682289455" --output knee_singlecoil_train.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xzAWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=XXBy9KH%2B65zzfFh62xWdn3a53ZU%3D&Expires=1682289455" --output knee_singlecoil_val.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xzAWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2Fo4xxDIrkuIc6M%2Fl%2Bgk0rVjLeg0%3D&Expires=1682289455" --output knee_singlecoil_test_v2.tar.xz
```
(link to fastMRI Dataset: <a href="https://fastmri.med.nyu.edu/"> Link </a>)

#  Clone this repo:
```bash 
https://github.com/noamkorngut/MRI_reconstruction.git

# set the project path in fastmri_dirs.yaml and data path in DataLoader_kspace.py
```
# for training run:
```bash 
python src/model train.py --model_name [KIKI, UNet] --loss_fn [MSE, L1]
```
# for evaluation run: 
```bash 
# set checkpoint path in the file and then run:
python src/model_evaluation.py --model_name [KIKI, UNet] 
```

# for part 3 please refer the ReconFormer directory
