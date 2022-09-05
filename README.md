# Diffusion Models for 2D Examples
This Repo contains the notebooks for the __Challenge 1: Score-based generative models: Implementation, optimisation, generalisation__ of the _Accelerating generative models and nonconvex optimisation_  workshop, which can be found here https://akyildiz.me/tmcf/challenges.html

## Setting Up the Environment
```
conda create --name sgm
conda activate sgm
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=sgm
pip install --upgrade "jax[cpu]"
pip install matplotlib
pip install seaborn
```

## Research Goals
### Generalization vs. Sample Density
#### Sphere
Given a sphere in d dimensions. Given a covering density epsilon. Use a uniform covering of the sphere as training examples and train a neural network. Optimally the final distribution would be the uniform distribution on the sphere. Find some good distance measure (just some heuristics are enough for first maybe) to measure the distance towards the training dataset + measure the distance towards the uniform distribution on the sphere. Compare how these behave w.r.t. d and epsilon.
One can also plot the distances of the trained drift to the true drift and to the trained drift, at least in low dimensions.

#### Hyperplane
For a hyperplane and a Gaussian distribution one can explicitely calculate the right solution at every time t. Given n samples, plot the distance of the trained drifts to the true drifts and the empirical drifts. Also heuristically plot the distance between the learned distribution to the data distribution / true distribution. So very similar as above.

Also for both of the above, its interesting to just maybe plot how much mass actually lands on the manifold

#### Explicit Score Matching

Instead of using denoising Score matching for the NN training one could study the effects of using explicit score matching instead.
