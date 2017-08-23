Generator and discriminator architecture is taken from:
SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient Paper
(link: https://arxiv.org/abs/1609.05473 , code: https://github.com/LantaoYu/SeqGAN)
Generator: LSTM
Discriminator: CNN

Two types of training are tested:
Policy Gradient [reference to paper: https://arxiv.org/abs/1609.05473]
Wasserstein GAN [reference to paper: https://arxiv.org/abs/1704.00028]


There are two folder PolicyGradientTraining and WassersteinGAN_Loss containing two different training methods with the same generator and discriminator architectures.

Both folders contain readme files about how to run the codes.
