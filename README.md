# ICML 3519: Generalization Bounds for Graph Neural Networks

This repo holds the supporting code for ICML submission 3519.

\textsc{Abstract.} We study the sample complexity of learning in graph neural networks by providing improved bounds on the Rademacher complexity. Previous work limited their analysis to a specific layer type and learning task. In contrast, we generalize our results to a variety of layer architectures and loss functions. Our complexity bounds are adaptive to network architectures and have improved dependence on the network depth. Under additional assumptions, we further show full independence of the network depth and width. We  provide empirical evidence to support our results and discuss the implications of our derivations for training graph neural networks.

Reproduce our results by running for example:
``python graphregression.py --batch_size 512 --learning_rate 2e-3 --epochs 30 --num_samples 100000``
