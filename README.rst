Computational Intelligence ToolKit (CITK)
========================================

This is an ultimate package for SOTA CI algorithmes

Installation
============

``git clone https://github.com/tupoylogin/Neural_Net_Genetic_Alg.git``

``cd Neural_Net_Genetic_Alg``

``pip install .`` (or ``pip install -e .`` to enable edit mode)

Examples
========

-  `Multilayer Perceptron trained with Genetic
   Algorithm <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GeneticAlgorithm.ipynb>`__
-  `Multilayer Perceptron trained with
   SGD <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/BackPropogationSGD.ipynb>`__
-  `Multilayer Perceptron trained with Genetic Algorithm and then with
   SGD <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GeneticAndSGD.ipynb>`__
-  `Multilayer Perceptron trained with Conjugate
   SGD <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/BackPropogationConjugateSGD.ipynb>`__
-  `ANFIS Neural Net trained with
   SGD <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/AnfisSGD.ipynb>`__
-  `GroupedMethod+Dense Neural Net with Layer Hypersearch trained with 
   SGD <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GMDHandDenseOnSGD.ipynb>`__
-  `GMDH Neural Net with batching on 
   LSM <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GMDH.ipynb>`__
-  `Fuzzy GMDH Neural Net with batching on 
   Linear Programming <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/FuzzyGMDH.ipynb>`__
-  `GMDH Neural Net on LSM on 
   Time Series data <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/GMDH_GDP.ipynb>`__
-  `Fuzzy GMDH Neural Net on Linear Programming on 
   Time Series data <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/FuzzyGMDH_GDP.ipynb>`__

Result Table
============

All experiments are carried on `Boston
dataset <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html>`__

Using such preprocessing: - Quantile Transform on Target
(``n_quantiles=300, output_distribution="normal"``) - Standard Scaling
of features

Test/Train splitting: - test size - 20% - use histogram bins
stratification

`Data preparation
code <https://github.com/tupoylogin/Neural_Net_Genetic_Alg/blob/main/examples/utils.py#L37>`__

Metric - MSE on normalized data

+-------------------------------------+---------------+--------------+
| Experiment name                     | Train score   | Test score   |
+=====================================+===============+==============+
| MLP+Genetic                         | 0.508         | 0.746        |
+-------------------------------------+---------------+--------------+
| MLP+SGD                             | 0.233         | 0.669        |
+-------------------------------------+---------------+--------------+
| MLP+(Genetic->SGD)                  | 0.244         | 0.636        |
+-------------------------------------+---------------+--------------+
| MLP+Conjugate SGD                   | 0.307         | 0.650        |
+-------------------------------------+---------------+--------------+
| ANFIS+SGD                           | 0.561         | 0.759        |
+-------------------------------------+---------------+--------------+
| GroupedMethod+Dense+SGD+LayerSearch | 0.191         | 0.386        |
+-------------------------------------+---------------+--------------+
| GMDH                                | 0.732         | 0.423        |
+-------------------------------------+---------------+--------------+
| FuzzyGMDH                           | 94.0          | 0.432        |
+-------------------------------------+---------------+--------------+

