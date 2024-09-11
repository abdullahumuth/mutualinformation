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

The similarity between the problem of calculating the conditional entropy and the problem of machine translation is recognized, and thus a very similar architecture that was laid out in Attention is All You Need<sup>[2](#attention)</sup> paper is used.

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
exp_mode = OrderedDict(:noise => 0.001)
hyper_params = OrderedDict(:gaussian_num => [0,32], :learning_rate => (10.0^x for x=-5:-1))

simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)

 ```

#### Parameter dictionary keys:

Below is a summary of the supported keys for the dictionaries that will go inside the simple_experiment function. 

noise=0, load = "", discrete=true, uniform = false, unique = false, fake = false, shuffle=false

- `data_gen_params`: From the transverse field Ising model, the parameters that will  are `L`, `J`, `g`, `t`, and `num_samples`.
- `exp_modes_params`: The parameters that was used to try out different modes of the experiment. The keys are listed below:

    - `noise=0`: The noise level that will be added to the wave function coefficients.

    - `load=''`: A special loading mode that will load the data from the sampled data in the `data/inputs` folder.

    - `uniform=false`: The boolean which determines whether to sample the spin states uniformly, regardless of the wave function coefficients.

    - `unique=false`: The boolean which determines whether to sample the spin states uniquely, regardless of the wave function coefficients. (Requires `uniform=true`)

    - `fake=false`: The boolean which determines whether to use an artificial mapping between the spin configurations and the wave function coefficients. 

    - `shuffle=false`: If the fake data is to be shuffled or not.

It should be noted that it's very easy with this setup to adapt this code to different methods to calculate the mutual information, different data generation methods, or to different ways to save the data by changing the `create_dataset` and `mutualinformation`, and the `save_results` functions.

### Plotting Results:
To plot the results of an experiment, use the following function:

```julia
custom_plot(configdict)
```


where `configdict` is a dictionary that has the following keys:
- `path`: The path to the folder that contains the results. The full relative path should be like this: `data/outputs/path/results/result.json`
- `x`: The name of the parameter that will be used as the x-axis in the plot.
- `y`: Names of the parameters that will be used as the y-axis in the plot.
- `other_vars`: The names of the parameters that will be used as the variable that will be used to create different lines in the same plot.
- `log_scale`: The variable determining whether the y-axis will be in log scale or not.

#### Example:

```julia
configdict = Dict("path" => "discrete_noise_results", "x" => "num_samples", "y" => ["entropy","conditional_entropy"], "other_vars" => ["noise"], "log_scale" => "y")
custom_plot(configdict)
```

### Generating dataset from a randomly initialized transformer:
To generate a dataset from a randomly initialized conditional entropy estimator model, use the following function:

```julia
generation_experiment(name, model_output_dim = 2, model_output_seq_len = 20, initially_generated_dim = 1, initially_generated_seq_len=2, num_samples = 10000; seed=313, get_data_from = "", discrete=true, conditional = true, kwargs...)
```

where:
-  `name` is the name of the experiment.

-  `model_output_dim` and `model_output_seq_len` are the dimensions of the output of the transformer.

- `initially_generated_dim` and `initially_generated_seq_len` are the dimensions of the data that will go inside the transformer.

- `num_samples` is the number of samples to be generated.

- `seed` is the seed for the random number generator.

- `get_data_from` is the path to the generated data (this is to only evaluate the already generated models, not to generate).

- `discrete` is the boolean that determines whether the generating model architecture will be discrete or not.

- `conditional` is the boolean that determines whether the data will be conditional or not (non-conditional data generation is not tested as it's not very useful).

#### Example:

```julia
generation_experiment("continuous_production_generation", 1, 2, 2, L, 2^16; seed=120, discrete=false, get_data_from = "", conditional = true)
```

### Other General Functions:
Alongside others, there are helper functions to modify the json outputs inside the "output_modifier.jl" file. These functions are used to modify the json outputs to be able to plot them in a more meaningful way.

As they are not so essential and mostly straightforward to understand, they are not explained here.

## 4. **Experiments and Results**
### a. Convergence test with different sample sizes of the discrete architecture

#### Experiment Setup:
The experiments were carried out for different configurations of the quantum system parameters and transformer hyperparameters:
- **System size (`L`)**: 20 spins
- **Coupling constant (`J`)**: -1
- **Transverse field strength (`g`)**: Varied across [-0.5, -1.0, -2.0]
- **Time (`t`)**: Varied between 0.0 and 1.0

#### Results:
The mutual information estimates varied across the different configurations. For example:
- When `g = -1.0` and `t = 0.1`, the mutual information was found to be **X**, with an entropy of **Y** and conditional entropy of **Z**.
- In cases where the transverse field strength `g` was weaker, the mutual information decreased, suggesting a lower complexity in learning the wave function.

### b. Convergence test with different sample sizes of the continuous architecture

### c. Time series analysis of mutual information estimates

### d. Effect of noise on mutual information estimates

### e. Learning rate sensitivity analysis

### f. Controlled experiments with generated data sampled from a randomly initialize transformer

### g. Controllable experiments with artificially generated mapping

### h. Shuffling test

### i. Uniform sampling test with different sample sizes

### j. Unique sampling test with different sample sizes




---

## 5. **Discussion and Limitations**
### Discussion:

#### Generalizability of the Method:
With this method of the mutual information estimation, the transformer model and the experiment running methods are ready to be adapted to different kinds of experiments in different types of topics, such as genetics, neuroscience, or any other field where metrics from information theory can be a useful tool.

#### Unique Condition of the Quantum System Data:
The task of finding the mutual information between a discrete and a continuous distribution is a challenging one. Which is even more challenging is that when the continuous random variable is a function of the discrete random variable, meaning that the continuous distribution is just delta functions at the points where the discrete distribution is non-zero. 

This is very different from the case that even though the discrete distribution is related to the continuous distribution, the continuous distribution is not a function of the discrete distribution and still a smooth distribution, like in the example that the discrete distribution is a DNA sequence and the continuous distribution is gene expression levels.

This might be the core cause in some limitations discussed below.

#### Computational Complexity:
The method sidelines the exponential burden of increasing system size, rather having a quadratic dependence, due to the transformer model. This is a significant advantage compared to some methods that are used to calculate the mutual information, such k-th nearest neighbors as in this article<sup>[3](#discretecontinuous)</sup>.

#### Satisfying the Goal of the Project:

In the context of computational quantum methods, goal of this project is to have a metric for how difficult it is to learn wave functions using neural quantum states (NQS). 

However, if unlike in the case of fully transverse magnetic field in the Ising model, the wave function coefficients are not the same, then there is essentially a one-to-one mapping between the spin configurations and the wave function coefficients. 

In this case, the conditional entropy $H(S| \Psi)$ is zero, and the mutual information $I(S;  \Psi)$ is equal to the entropy of the spin configurations $H(S)$. 

Assuming the above condition is met, which is true in the overwhelming majority of the cases, the mutual information will be the entropy of the spin states, _regardless of the clustering_ of the wave function coefficients. This is supposedly not what we want as a metric for the complexity.

Adding noise will change the theoretical value of the mutual information, but it might still be a good metric for the learning complexity of the system.

- It seems that in the discrete case, the conditional entropy calculation is heavily dependent on the number of samples, and in most cases it seems like it is decreasing without convergence.

### Limitations:
- **Shuffling**: 
    - When the spin state representations are shuffled, the mutual information is theoretically the same as before. However, in sufficiently large datasets where the network cannot memorize individual entries, the experiments show that when the network cannot figure out any pattern in the data, the estimated mutual information is zero.
    So, a pattern is needed for this method to work in large datasets. This can be a limitation of mutual information estimation in deep learning methods that have fixed number of parameters that are much less than the number of samples.

- **Convergence**: 
    - In the discrete architecture, the entropy calculation is converging to the correct value very easily.
    - In the discrete architecture, the conditional entropy calculation is heavily dependent on the number of samples, and in most cases it seems like it is decreasing without convergence. And in the cases where the conditional entropy is theoretically 0, the network does not seem to be able to learn this fact.
    - The continuous architecture is better in this regard, as the predicted mutual information mostly seems to converge to the entropy of the discrete case when the theoretical conditional entropy is 0. But for it to converge, it still requires a very large number of samples.

- **Training Process**: 
    - In the discrete case, with the termination condition of having no more improvement in the test loss for the last 500 epochs, with the increasing number of samples, the training process is getting longer for the conditional entropy estimation. For the entropy estimation, the training process is very stable and does not require a lot of epochs to converge.

    - Similarly, in the continuous case, the training process is getting even longer with the increasing number of samples, but unlike the discrete case, the training process seems to fluctiate a lot even with the entropy estimation, the loss graphs always seem to have a lot of spikes.

- **Computational Cost**: 
    - The system size increase can be tolerated up to GPU parallelization limits (and after that it's just a quadratic increase), but the number of samples increase the training time. This effect is more pronounced than just a proportional increase because the termination epoch is also increasing with the number of samples.

---

## 6. **Future Work**
1. **Working with fully known datasets**: 
    - With a randomly initialized conditional entropy model, the conditional entropy is known from the start. The entropy can be calculated through integration of the joint probability density. In this case, the mutual information is precisely known, which will allow rigorous validations of the method.
2. **Overcoming the delta function problem**: 
    - When the fact that the continuous random variable is a function of the discrete random variable is no longer true, the distrubution of the continuous random variable can be smoother, and the mutual information can be calculated more accurately and meaningfully. To achieve this, a method can be applying dropout to the embedding layer of the spin states, but more testing is needed to see if this will work. 
---

## 7. **References**
1. <a id="formallimitations"></a> McAllester, D., & Stratos, K. (2018, November 10). Formal limitations on the measurement of mutual information. arXiv.org. https://arxiv.org/abs/1811.04251
2. <a id="attention"></a> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017, June 12). Attention is all you need. arXiv.org. https://arxiv.org/abs/1706.03762
3. <a id="discretecontinuous"></a> Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357. https://doi.org/10.1371/journal.pone.0087357