# Stochastic Neural Simulator for Generalizing Dynamical Systems across Environments

Official code and supplimentary material for IJCAI 2024 paper Stochastic Neural Simulator for Generalizing Dynamical Systems across Environments

## Abstract

Neural simulators for modeling complex dynamical systems have been extensively studied for various real-world applications, such as weather forecasting, ocean current prediction, and computational fluid dynamics simulation. Although they have demonstrated powerful fitting and predicting, most existing models are only built to learn single-system dynamics.
Several advanced researches have considered learning dynamics across environments, which can exploit the potential commonalities among the dynamics across environments and adapt to new environments. However, these methods still are prone to scarcity problems where per-environment data is sparse or limited. Therefore, we propose a novel CoNDP (**Co**ntext-Informed **N**eural O**D**E **P**rocesses) to achieve learning system dynamics from sparse observations across environments. It can fully use contextual information of each environment to better capture the intrinsic commonalities across environments and distinguishable differences among environments while modeling uncertainty of system evolution, producing more accurate predictions. Intensive experiments are conducted on five complex dynamical systems in various fields. Results show that the proposed CoNDP can achieve optimal results compared with common neural simulators and state-of-the-art cross-environmental models.

