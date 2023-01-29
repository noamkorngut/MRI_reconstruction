# MRI_reconstruction



# Dataset:
# for train, val and test sets run the following commands:
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_train.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qV5z%2Bt5EVqoExVsS%2F%2B%2Fb6O4Tneg%3D&Expires=1682289455" --output knee_singlecoil_train.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=XXBy9KH%2B65zzfFh62xWdn3a53ZU%3D&Expires=1682289455" --output knee_singlecoil_val.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2Fo4xxDIrkuIc6M%2Fl%2Bgk0rVjLeg0%3D&Expires=1682289455" --output knee_singlecoil_test_v2.tar.xz

#set the project path in fastmri_dirs.yaml and dath path in DataLoader_kspace.py

#for training run:
model train.py --model_name [KIKI, UNet] --loss [MSE, L1]

#for evaluation run: (set checkpoint path in the file)
model_evaluation.py --model_name [KIKI, UNet] 
