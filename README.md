# Mitigating Generation Shifts for Generalized Zero-Shot Learning (PyTorch Implementation) （ACM Multimedia 2021)


Abstract: Generalized Zero-Shot Learning (GZSL) is the task of leveraging semantic information (e.g., attributes) to recognize the seen and unseen samples, where unseen classes are not observable during training. It is natural to derive generative models and hallucinate training samples for unseen classes based on the knowledge learned from seen samples. However, most of these models suffer from the generation shifts, where the synthesized samples may drift from the real distribution of unseen data. In this paper, we conduct in-depth analysis on this issue and propose a novel Generation Shifts Mitigating Flow (GSMFlow) framework, which is comprised of multiple conditional affine coupling layers for learning unseen data synthesis efficiently and effectively. In particular, we identify three potential problems that trigger the generation shifts for this task, i.e., semantic inconsistency, variance decay, and structural permutation and address them respectively. First, to reinforce the correlations between the generated samples and its respective attributes, we explicitly embed the semantic information into the transformations in each of the coupling layers. Second, to recover the intrinsic variance of the synthesized unseen features, we introduce a visual perturbation strategy to diversify the intra-class variance of generated data and hereby help adjust the decision boundary of classifier. Third, to avoid structural permutation in the semantic space, we propose a relative positioning strategy to manipulate the attribute embeddings, guiding which to fully preserve the inter-class geometric structure. Experimental results demonstrate that GSMFlow is capable of generating reliable unseen data points and achieves the state-of-the-art recognition performance in both classic and generalized zero-shot settings.

# Requirements

FrEIA pip install git+https://github.com/VLL-HD/FrEIA.git@550257b10af7d8772b08d4aa9b18772e2c02 

Python 3.8

torch 1.6

Numpy 1.19.5

Sklearn 0.24.1

Scipy 1.6

# Usage

Put your [datasets](https://www.dropbox.com/s/5sprhvavkb3pu5m/GSMFlow_data.zip?dl=0) in data folder and run the scripts:

**AWA2**:
```
python train_AWA2.py
```
**APY**:
```
python train_APY.py
```
**CUB**:
```
python train_CUB.py
```
**FLO**:
```
python train_FLO.py
```