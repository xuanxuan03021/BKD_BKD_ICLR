# Backdoor Defense via Adaptively Splitting Poisoned Dataset

This repository provides the pytorch implementatin of our work:

## Abstract
Backdoor attacks pose significant security risks to deep neural networks (DNNs) by manipulating model predictions through training dataset poisoning. To mitigate backdoor threats, \textit{anti-backdoor learning} has attracted increasing interest, aiming to train clean models directly from poisoned datasets.  
However, existing methods usually fail to recover backdoored samples to their original, correct labels and suffer from poor efficiency and dependency on precise backdoor isolation. To address these issues, we first explore the fundamental differences between training a poisoned model and a clean model on a poisoned dataset from a causal perspective. Our theoretical causal analysis reveals that incorporating \emph{\textbf{both}} images and the associated attack indicators preserves the model's integrity. Specially, we use frequency spectrum of the image as the indicator of attack. Building on this insight, we introduce an end-to-end method, Mind Control through Causal Inference (MCCI), to mitigate backdoors. This approach consists of a Semantic Feature Recognition Network (SFRN) that learns semantic information of images, and an Attack Indication Network (AIN) that controls the model's perception of whether an input is clean or poisoned based on the frequency spectrum. In the inference stage, we control the model's perception by feeding AIN with a frequency spectrum from clean images, ensuring that all inputs are perceived as clean. 
Extensive experiments demonstrate that our method achieves state-of-the-art performance, efficiently recovering the original correct predictions for poisoned samples and enhancing accuracy on clean samples.



## Installation

This code is tested on our local environment (python=3.7, cuda=11.1), and we recommend you to use anaconda to create a vitural environment:

Activate the environment:
```bash
conda env create -f MCCI_ENV.yml
```


## Backdoor Defense

Run the following command to train our ASD under BadNets attack.

```shell
python test_bkd_bkd_BadNet.py 
```

We provide a pretrained models (./BackdoorBackdoor_BadNet_0.1_100.pth)

If you wish to train your own model, you can delete this and it will automatedly train a new one.

