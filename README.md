# DataAugmentation__NLP
Generating text containing negative sentiment


-Experiment 1
cd Experiment1

to preprocess text:
python text_generator.py

Generator: LSTM and Discriminator: CNN
Training method: Wasserstein Training
python wasserstein_GAN.py

Training method: Policy Gradient
python policyGradient.py

-Experiment 2
cd Experiment2

to preprocess text:
python text_generator.py

Generator: LSTM and Discriminator: LSTM
Training method: Policy Gradient

python sequence_gan_vs2.py

-Experiment 3
cd Experiment3

to preprocess text:
python text_generator.py

Generator: LSTM and Discriminator: LSTM
Training method: Policy Gradient + Teacher Forcing

python sequence_gan_teacher_forcing.py



>>>Folder 'Evaluation'

contains scripts to compute different evaluation metrics.



>>>>>>>>>>>>>>>>>>>>>>>>>>

Implementation references:

Improved training for wasserstein gan:
https://github.com/igul222/improved_wgan_training

policy gradient:
https://github.com/LantaoYu/SeqGAN

>>>>>>>>>>>>>>>>>>>>>>>>>

