#!/bin/env bash
# ReconFormer Evaluation
python main_recon_test.py --phase test --model ReconFormer --challenge singlecoil --F_path 'path to fastMRI dataset' --test_dataset F --sequence PD --accelerations 8 --center-fractions 0.04 --checkpoint 'path to checkpoint'--verbose
# python main_recon_test.py --phase test --model ReconFormer --challenge singlecoil --F_path /tcmldrive/Noam/AIsap/MRI_reconstruction --test_dataset F --sequence Dataset --accelerations 4 --center-fractions 0.04 --checkpoint /tcmldrive/Noam/AIsap/MRI_reconstruction/ReconFormer/checkpoints/F_X4_checkpoint.pth  --verbose
