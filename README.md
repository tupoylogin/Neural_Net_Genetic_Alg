# Computational Intelligence Toolkit (CIT)

This is an ultimate package for SOTA CI algorithmes 

# Installation

`git clone https://github.com/tupoylogin/Neural_Net_Genetic_Alg.git`

`cd Neural_Net_Genetic_Alg`

`pip install .` (or `pip install -e .` to enable edit mode)

# Examples

- [Multilayer Perceptron trained with Genetic Algorithm](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GeneticAlgorithm.ipynb)
- [Multilayer Perceptron trained with SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/BackPropogationSGD.ipynb)
- [Multilayer Perceptron trained with Genetic Algorithm and then with SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GeneticAndSGD.ipynb)
- [Multilayer Perceptron trained with Conjugate SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/BackPropogationConjugateSGD.ipynb)
- [ANFIS Neural Net trained with SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/AnfisSGD.ipynb)
- [GMDH Neural Net with Layer Hypersearch trained with SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GMDHandDenseOnSGD.ipynb)
- [Fuzzy GMDH Neural Net with Layer Hypersearch trained with SGD](https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/FuzzyGMDHandDenseOnSGD.ipynb)

# Result Table

|Exepriment name                 |Train score                     |Test score                      |
|--------------------------------|:------------------------------:|:------------------------------:|
|MLP+Genetic                     |0.455                           |0.645                           |
|MLP+SGD                         |0.323                           |0.590                           |
|MLP+(Genetic->SGD)              |0.284                           |0.558                           |
|MLP+Conjugate SGD               |0.367                           |0.563                           |
|ANFIS+SGD                       |0.621                           |0.768                           |
|GMDH+SGD                        |0.191                           |0.386                           |
|FuzzyGMDH+SGD                   |0.459                           |0.628                           |
