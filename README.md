# Nonparametric Off-Policy Policy Gradient


<div align="center">

<img src="img/nopg.png" alt="Paris" style="display: block; margin-left: auto; margin-right: auto; width: 30%;">
</div>

<br>

[Nonparametric Off-Policy Policy Gradient](https://arxiv.org/abs/2001.02435) (NOPG) is a Reinforcement Learning algorithm for off-policy datasets. The gradient estimate is computed in closed-form by modelling the transition probabilities with Kernel Density Estimation (KDE) and the reward function with Kernel Regression.

The current version of NOPG supports stochastic and deterministic policies, and works for continuous state and action spaces. An extension to discrete spaces will be made available in the near future.

It supports environments with openAI-gym like interfaces.

Link to arXiv: [https://arxiv.org/abs/2001.02435](https://arxiv.org/abs/2001.02435)

## Install

The code was tested with Python 3.6.8 in a machine with Ubuntu 18.04 and uses PyTorch for automatic gradient computation. We recommend using a GPU and large RAM to improve the training speed.

Install all dependencies by running

```bash
bash setup.sh
```


## Run

The easiest way to create an experiment is to follow the template in [examples/template.py](examples/template.py) or directly look at the examples in the [examples](examples) directory.


## Example

**Swing-up Pendulum with Uniformly sampled dataset and Deterministic Policy**

Run the code with 
```python
python examples/pendulum_nopg_d_uniform.py
```
You should get roughly a non-discounted return close to -500.

