# Stochastic Neural Simulator for Generalizing Dynamical Systems across Environments

Official code and supplimentary material for IJCAI 2024 paper Stochastic Neural Simulator for Generalizing Dynamical Systems across Environments

##Abstract
Neural simulators for modeling complex dynamical systems have been extensively studied for various real-world applications, such as weather forecasting, ocean current prediction, and computational fluid dynamics simulation. Although they have demonstrated powerful fitting and predicting, most existing models are only built to learn single-system dynamics.
Several advanced researches have considered learning dynamics across environments, which can exploit the potential commonalities among the dynamics across environments and adapt to new environments. However, these methods still are prone to scarcity problems where per-environment data is sparse or limited. Therefore, we propose a novel CoNDP (\underline{Co}ntext-Informed \underline{N}eural O\underline{D}E \underline{P}rocesses) to achieve learning system dynamics from sparse observations across environments. It can fully use contextual information of each environment to better capture the intrinsic commonalities across environments and distinguishable differences among environments while modeling uncertainty of system evolution, producing more accurate predictions. Intensive experiments are conducted on five complex dynamical systems in various fields. Results show that the proposed CoNDP can achieve optimal results compared with common neural simulators and state-of-the-art cross-environmental models.

##Model Overview
CoNDP consists of four main components: 
an initial encoder that infers the initial latent states of the system;
a context-informed encoder for learning the environment representation, where the attention weights consist of shared weights and environment-specific weights determined by environment-specific parameters and a hyper network;
a process of solving initial value problems in hidden space;
and a decoder for predicting future system states. 
Initial encoder estimates the distribution of initial latent state $\bm{z}_{t_0}$ from initial state $\bm{x}_{t_0}^e$, i.e., $p(\bm{z}_{t_0}|\bm{x}_{t_0}^e)$. 
We denote the historical trajectories over the past $K$ timestamps as $Context=(\bm{x}_{t_{-K}}^e,\cdots,\bm{x}_{t_{-1}}^e)$.
The context-informed encoder is to map the $Context$ into a conditional distribution of environmental representation $\bm{u}^e$, i.e., $p(\bm{u}^e|Context, \bm{\xi}^e)$, which is determined by environment-specific parameters $\bm{\xi}^e$ and a hyper network.
The environmental representation can be viewed as a random control vector of the system dynamics, which is fed into a control net to produce the environment-specific component of the governing equation $f$.
By sampling the initial latent state $\bm{z}_{t_0}$, we can solve the IVP to generate the hidden state at any time $t$, i.e., $\bm{z}_{t}$.
Then, a decoder is used to provide the predictive distribution $p(\bm{x}_{1:T}^e|\bm{z}_{t_0}, \bm{u}^e)$.
On the contrary, all shared parameters and environment-specific parameters are learned based on the empirical error between predicted and true values.
