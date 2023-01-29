# MRI_reconstruction



## Dataset:
Prepare the dataset in the following structure for easy use of the code.The provided data loaders is ready for this this format and you may change it as your need.


```bash


                  
                  |                       
                  |                |--xxx.h5  
                  |--train--|...
                  |                |...
 Dataset-----|    |                   
                  |
                  |                |--xxx.h5      
                  |--val -|...
                  |                |...  
                              
                          
                                    
 ```

## for train, val and test sets downloading run the following commands:
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_train.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qV5z%2Bt5EVqoExVsS%2F%2B%2Fb6O4Tneg%3D&Expires=1682289455" --output knee_singlecoil_train.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=XXBy9KH%2B65zzfFh62xWdn3a53ZU%3D&Expires=1682289455" --output knee_singlecoil_val.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2Fo4xxDIrkuIc6M%2Fl%2Bgk0rVjLeg0%3D&Expires=1682289455" --output knee_singlecoil_test_v2.tar.xz

(link to fastMRI Dataset: <a href="https://fastmri.med.nyu.edu/"> Link </a>)
#set the project path in fastmri_dirs.yaml and dath path in DataLoader_kspace.py

## Clone this repo:
```bash 
https://github.com/noamkorngut/MRI_reconstruction.git
```
##for training run:
```bash 
model train.py --model_name [KIKI, UNet] --loss [MSE, L1]
```
##for evaluation run: (set checkpoint path in the file)
```bash 
model_evaluation.py --model_name [KIKI, UNet] 
```
