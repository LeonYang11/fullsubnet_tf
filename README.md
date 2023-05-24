# Tensorflow implementation of fullsubnet for noise suppression
This repository is the unofficial implementation of paper "FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement" by tensorflow 1.X
#
Basic training scripts of DPCRN (https://github.com/Le-Xiaohuai-speech/DPCRN_DNS3) are used here
# Requirements
tensorflow>=1.14
# Train and Test
python main_fullsubnet.py --mode train --cuda 0 --experimentName exp0
#
python main_fullsubnet.py --mode test --test_dir the_dir_of_noisy --output_dir the_dir_of_enhancement_results
