# Estimating Mutual Information Between Distributions of Spin States and Wave Function Coefficients Using Transformers

This is a project report of my summer internship project at the [Computational Quantum Science Research Group](https://www.computational-quantum.science/), part of the [Institute of Quantum Control](https://www.fz-juelich.de/en/pgi/pgi-8) at the Forschungszentrum Jülich, Germany, 2024.

I would like to thank Dr. Markus Schmitt, the leader of the research group, for his helps through the internship period and before, and for providing me with the opportunity to work on this project.

Abdullah Umut Hamzaoğulları, September 2024

## 1. **Topic**
This project investigates the estimation of mutual information between distributions of quantum spin states and wave function coefficients.


### Project Goal
The goal is to develop a metric for how difficult it is to learn wave functions using neural quantum states (NQS). The study is motivated by the need to better understand how neural networks can efficiently represent many-body quantum systems.

### Problem Statement
Mutual information between spin configurations and wave function coefficients can serve as a useful proxy for determining the complexity of representing quantum states using machine learning techniques. This project aims to estimate this mutual information for various many-body quantum systems, by difference of entropies method<sup>[1](#formallimitations)</sup> using transformer networks with Monte Carlo sampling.

---

## 2. **Approach**
### Methodology

#### Mutual Information Estimation
Two separate transformer architectures described below are used to estimate entropy and conditional entropy between two sets of variables (spin configurations and wave function coefficients). The difference between these two gives an estimate of the mutual information:

$$I(S;  \Psi) = H(S) - H(S| \Psi)$$
or equivalently:
$$I(S;  \Psi) = H( \Psi) - H( \Psi|S)$$
where $S$, a discrete distribution, represents the spin configurations and $\Psi$, a continuous distribution, represents the corresponding wave function coefficients.

To expand the first equation:

$$H(S) = - \sum_{i} p(S=s_i) \log p(S=s_i)$$

and

$$H(S| \Psi) = - \sum_{i} \int \rho(S=s_i,\Psi=\psi) \log p(S=s_i| \Psi=\psi) d\psi$$

where $p(S=s_i)$ is the probability of the i-th spin configuration, $\rho( \Psi=\psi,S=s_i)$ is the joint probability density of the continuous and discrete distributions, and $\psi$ is a complex number, integrated over all possible values of $\Psi$.


Rather than calculating the value of the joint probability density $\rho( \Psi=\psi,S=s_i)$ occuring inside the integral, Monte Carlo sampling which is drawn from that distribution is used.

So,

$$H(S) = - \sum_{i} \frac{1}{N} \log p(S=s_i)$$

$$H(S| \Psi) = - \sum_{s_i,\psi_i} \frac{1}{N} \log p(S=s_i| \Psi=\psi_i)$$

where $N$ is the number of samples drawn from the distribution, and
 $$s_i,\psi_i \sim \rho( \Psi=\psi,S=s_i)$$

So, the entropy and conditional entropy are just negative log-likelihoods of the entropy and conditional entropy of the distributions, respectively.



##### A note on the continuous formulation
One thing to note is that in the second equation, the entropy of the continuous distribution is calculated using the differential entropy formula, which is the continuous analog of the discrete entropy formula:

$$H( \Psi) = - \int \rho( \Psi) \log \rho( \Psi) d \Psi$$

and 

$$H( \Psi|S) = - \int \sum_{i}\rho( \Psi,S) \log \rho( \Psi|S) d\Psi$$

<br/> 
More explicitly, the entropies are calculated as follows:

$$H( \Psi) = - \int \rho( \Psi=z) \log \rho( \Psi=z) dz$$

and

$$H( \Psi|S) = - \int \sum_{i}\rho( \Psi=z,S=s_i) \log \rho( \Psi=z|S=s_i) dz$$

where $\rho( \Psi)$ and $\rho( \Psi|S)$ are the probability densities of the continuous distribution, $s_i$ is the i-th spin configuration, and $z$ is a complex number, integrated over all possible values of $\Psi$.


In these cases, even though these differential entropies are not well-defined and can produce negative values when the probabilty density $\rho$ is above 1, the difference between them is well-defined and is the mutual information between the two distributions.



#### Data Generation
 The input data consists of quantum spin configurations and their corresponding wave function coefficients. Data is derived through Monte Carlo sampling of the spin states and the corresponding wave function coefficients of the transverse-field Ising model.


#### Model Architecture
The transformer architecture for the entropy estimation is a decoder-only transformer, while the architecture for conditional entropy estimation is a standard transformer with encoder and decoder components. 

The similarity between the problem of calculating the conditional entropy and the problem of machine translation is recognized, and thus a very similar architecture that was laid out in "Attention is All You Need" paper is used.

In the discrete case, the transformer is trained to predict the probability of each spin configuration, auto-regressively. Then the entropy and the conditional entropy is calculated using the negative log-likelihood of the predicted probabilities.

In the continuous case, the transformer is trained to predict the underlying continuous distribution that single wave function coefficient belongs to, using a mixture of Gaussians. Then, the entropy and the conditional entropy is calculated using the negative log-likelihood of the predicted probability densities. 

The project's main computational task is to run several experiments with different system parameters (such as `L`, `J`, `g`, and `t`) and model hyperparameters (such as number of Gaussian components in the transformer).



## 3. **How to Use the Code**
### Requirements:
To run the project, ensure that the following packages are installed:


- `Flux`
- `Transformers`
- `CUDA`
- `NeuralAttentionlib`
- `ProgressMeter`
- `Random`
- `Statistics`
- `Distributions`
- `Plots`
- `OrderedCollections`
- `DataFrames`
- `BSON`
- `CSV`
- `HDF5`
- `JSON3`


### Installation:
1. Clone the repository:
    ```bash
    git clone https://github.com/abdullahumuth/mutualinformation.git
    cd <repository-directory>
    ```

2. Set up Julia environment and activate it:
    ```bash
    julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

### Running Experiments:
To run a simple experiment, use the following function:

```julia
simple_experiment(
    experiment_name, 
    version_number, 
    data_gen_params; 
    exp_modes_params, 
    hyper_parameters)
```

where the `data_gen_params`, `exp_modes_params`, `hyper_parameters` are dictionaries whose keys are parameter names (as symbols in Julia) and the values are any iterable set of values for the experiment to go over. 

This function call will:
- Create necessary directories.
- Run the experiment with different configurations of the parameters (as their cartesian product)
- Save models, plots, and results in the `data/outputs/experiment_name` folder.


Inside the loop of parameters, input is created through `create_dataset` function, which takes `data_gen_param_dict` and `exp_modes_param_dict` as arguments, which are NamedTuples, and can be used to pass the parameters with the ... operator.

```julia
input = create_dataset(data_gen_param_dict...; exp_modes_param_dict...)
```

The `create_dataset` function returns a tuple, and then is used to calculate the mutual information by:
    
```julia
entropy, conditional_entropy = mutualinformation(input...; hyper_param_dict...)
```

Where `hyper_param_dict` is a NamedTuple that contains the hyperparameters of the model.

Then, with the entropy and conditional entropy known, the models, loss csv and plots, and the results are saved with:

```julia
save_results(entropy, conditional_entropy, name, data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
```

Where `name` is the name of the experiment that will be used to create the folder structure in the `data/outputs` folder.



It's very easy with this setup to adapt this code to different methods to calculate the mutual information, or to different data generation methods.

#### Example:
    

A complete example of running an experiment from `experiment3.jl` is below:

 ```julia
include("./main.jl")
include("plot_from_json.jl")
name = "learning_rate_tests"
version = 1

L = [20, 12]
J = -1
g = -1.0
t = 0.1

data_gen_params = OrderedDict(:L => L, :J => J, :g => g, :t => 0.1:0.1:0.9, :num_samples=>(2^x for x=5:16))
exp_mode = OrderedDict()
hyper_params = OrderedDict(:gaussian_num => [0,32], :learning_rate => (10.0^x for x=-5:-1))

simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)

 ```

#### Parameters






## 4. **Experiments and Results**
### Experiment Setup:
The experiments were carried out for different configurations of the quantum system parameters and transformer hyperparameters:
- **System size (`L`)**: 20 spins
- **Coupling constant (`J`)**: -1
- **Transverse field strength (`g`)**: Varied across [-0.5, -1.0, -2.0]
- **Time (`t`)**: Varied between 0.0 and 1.0

### Results:
The mutual information estimates varied across the different configurations. For example:
- When `g = -1.0` and `t = 0.1`, the mutual information was found to be **X**, with an entropy of **Y** and conditional entropy of **Z**.
- In cases where the transverse field strength `g` was weaker, the mutual information decreased, suggesting a lower complexity in learning the wave function.

Plots of the training losses for entropy and conditional entropy were generated and saved in the `data/outputs` folder.

---

## 5. **Discussion and Limitations**
### Discussion:
The results indicate that the complexity of learning quantum states using transformer networks is highly dependent on the system parameters. For stronger transverse field strengths, the mutual information increased, implying that the quantum state becomes more complex to learn. Additionally, the transformer model was able to successfully approximate the entropy and conditional entropy, though performance varied based on hyperparameter tuning.

### Limitations:
- **Scalability**: While transformers provide flexibility in modeling complex distributions, the computational cost grows significantly with system size `L`, and deeper networks are required to capture subtle features of the wave functions.
- **Accuracy of Monte Carlo Sampling**: The accuracy of the estimated mutual information relies heavily on the Monte Carlo sampling process. More sophisticated sampling techniques or larger sample sizes might be required to reduce variance in the estimates.
- **Hyperparameter Sensitivity**: The results are sensitive to transformer-specific hyperparameters such as the number of Gaussian components used. More experiments are needed to robustly tune these parameters.

---

## 6. **Future Work**
1. **Improved Sampling Techniques**: Exploring more advanced Monte Carlo methods or using variational techniques to sample from quantum distributions could improve the accuracy of mutual information estimates.
2. **Larger System Sizes**: Extending the analysis to larger quantum systems (increasing `L` beyond 20) will help test the scalability of the method.
3. **Comparison with Other Machine Learning Models**: Future work could involve comparing transformer performance with other neural architectures, such as Restricted Boltzmann Machines (RBMs) or convolutional networks, for learning quantum states.
4. **Incorporation of More Quantum Systems**: Extending the analysis to other quantum systems (e.g., Hubbard models or fermionic systems) could generalize the applicability of the transformer-based approach.

---

## 7. **References**
1. <a id="formallimitations"></a> McAllester, D., & Stratos, K. (2018, November 10). Formal limitations on the measurement of mutual information. arXiv.org. https://arxiv.org/abs/1811.04251
2. Formal Limitations on the Measurement of Mutual Information, Paper Reference [link to the paper].
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI GPT-2*.
