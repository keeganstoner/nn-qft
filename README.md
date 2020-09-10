# Neural Networks and Quantum Field Theory

Implementation of 

https://arxiv.org/abs/2008.08601

by James Halverson, Anindita Maiti, and Keegan Stoner

## Abstract

We propose a theoretical understanding of neural networks in terms of Wilsonian effective field theory. 
The correspondence relies on the fact that many asymptotic neural networks are drawn from Gaussian processes, the analog of non-interacting field theories. 
Moving away from the asymptotic limit yields a non-Gaussian process and corresponds to turning on particle interactions,
allowing for the computation of correlation functions of neural network outputs with Feynman diagrams.
Minimal non-Gaussian process likelihoods are determined by the most relevant non-Gaussian terms, according to the flow in their coefficients induced by the Wilsonian renormalization group. This yields a direct connection between overparameterization and simplicity of neural network likelihoods.
Whether the coefficients are constants or functions may be understood in terms of GP limit symmetries,
as expected from 't Hooft's technical naturalness.
General theoretical calculations are matched to neural network experiments in the simplest class of models allowing the correspondence.
Our formalism is valid for any of the many architectures that becomes a GP in an asymptotic limit, a property preserved under
certain types of training.


## Using this code

```python script.py --activation=ReLU --d-in=1 --n-models=10**6```

runs script.py with 10**6 models using the ReLU activation for the 1D inputs listed in Section 2. 

### 1. ```generate_models.py```

This code generates models for a given activation function and input dimension and saves the outputs in a pickle file. These can be used in the other scripts where the analysis is done.

This generates the models and outputs them as pickle files that are used in the other three scripts.

The options are:

- ```--activation``` - activation function defining network architecture
    - ```ReLU, Erf, GaussNet```
- ```--d-in``` - input dimension of xs
    - ```1, 2, 3```
- ```--n-models``` - total number of models created/used in each experiment
- ```--mw``` - mean of W distribution (default = 0.0)
- ```--mb```  - mean of b distribution (default = 0.0)
- ```--sw``` - standard deviation of W distribution (default = 1.0)
- ```--sb```- standard deviation of b distribution (default = 1.0, except ReLU sets to 0.0)


### 2. ```free_theory.py```

This inhereits the options from the previous code but adds
- ```--n-pt```- plots either the 2-, 4-, or 6-pt function for ```activation```
    - ```2, 4, 6```

Uses the models to show falloff of 4-pt and 6-pt signals to their GP predictions as a function of width, as well as 2-pt signals below background noise level as shown in Section 2.

### 3. ```rg.py```

Uses the models to show the 4-pt coupling lambda change with varying cutoff scale in accordance with the RG equations in Section 4. Also pickles the lambda tensor at each cutoff for use in ```six_pt_prediction.py```.

### 4. ```six_pt_prediction.py```

Uses the models and the lambda tensor from ```rg.py``` to predict the 6-pt function for ```activation``` at the NGP width used in the paper.




## Contact

**Code authors:** Keegan Stonerm Anindita Maiti<p> 
**Issues and questions:** @keeganstoner, stoner.ke@northeastern.edu <p>


## BibTeX Citation
``` 
@misc{halverson2020neural,
    title={Neural Networks and Quantum Field Theory},
    author={James Halverson and Anindita Maiti and Keegan Stoner},
    year={2020},
    eprint={2008.08601},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```