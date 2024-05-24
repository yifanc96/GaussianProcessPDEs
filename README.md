# GaussianProcessPDEs.jl
This is an integrated repo for using Gaussian processes to solve PDEs, learn PDEs, solve inverse problems, for uncertainty quantification, parameter estimation as well as fast multiscale algorithms for computation with dense kernel matrices.

We use Python (especially JAX, for automation and ease of code writting) and Julia (for fast algorithms and detailed speed up). Most are basic notebooks in this repo. For other more advanced uses please refer the relevant repos list below.

[Note] Files in "others" folders are not cleaned

## Relevant repositories:
* [NonLinPDEs-GPsolver](https://github.com/yifanc96/NonLinPDEs-GPsolver): Python-JAX for solving and learning PDEs using GPs
* [Time-Dependent-PDEs-GPsolver](https://github.com/yifanc96/Time-Dependent-PDEs-GPsolver): Python-JAX for solving time dependent PDEs
* [HighDimPDEs-GPsolver](https://github.com/yifanc96/HighDimPDEs-GPsolver): Python-JAX for solving high dimensional PDEs and parametric PDEs
* [PDEs-GP-KoleskySolver](https://github.com/yifanc96/PDEs-GP-KoleskySolver): Julia for near-linear time complexity multiscale Cholesky algorithm of GPs solving PDEs

## Relevant papers:
1. Yifan Chen, Bamdad Hosseini, Houman Owhadi, Andrew M. Stuart. "[Solving and Learning Nonlinear PDEs with Gaussian Processes](https://arxiv.org/abs/2103.12959)", Jounal of Computational Physics, 2021.

2. Yifan Chen, Houman Owhadi, Andrew M. Stuart. "[Consistency of Empirical Bayes And Kernel Flow For Hierarchical Parameter Estimation](https://arxiv.org/abs/2005.11375), Mathematics of Computation, 2021.

3. Yifan Chen, Houman Owhadi, Florian Schaefer. "[Sparse Cholesky Factorization for Solving Nonlinear PDEs via Gaussian Processes](https://arxiv.org/abs/2304.01294)", Mathematics of Computation, 2024.

4. Pau Batlle, Yifan Chen, Bamdad Hosseini, Houman Owhadi, Andrew M. Stuart. "[Error Analysis of Kernel/GP Methods for Nonlinear and Parametric PDEs](https://arxiv.org/abs/2305.04962)", arXiv preprint, 2023.

5. Yifan Chen, Bamdad Hosseini, Houman Owhadi, Andrew M Stuart. "[Gaussian Measures Conditioned on Nonlinear Observations: Consistency, MAP Estimators, and Simulation](https://arxiv.org/abs/2405.13149)", arXiv preprint, 2024. 


Some of the analytic results for multiscale PDEs are also related to the following works

5. Yifan Chen, Thomas Y. Hou. "[Multiscale Elliptic PDEs Upscaling and Function Approximation via Subsampled Data](https://arxiv.org/abs/2010.04199)", SIAM Multiscale Modeling and Simulation, 2022.

6. Yifan Chen, Thomas Y. Hou. "[Function Approximation via The Subsampled Poincare Inequality](https://arxiv.org/abs/1912.08173)", Discrete & Continuous Dynamical Systems - A, 2020.
