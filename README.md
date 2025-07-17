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
```
@article{chen2021solving,
  title={Solving and learning nonlinear PDEs with Gaussian processes},
  author={Chen, Yifan and Hosseini, Bamdad and Owhadi, Houman and Stuart, Andrew M},
  journal={Journal of Computational Physics},
  volume={447},
  pages={110668},
  year={2021},
  publisher={Elsevier}
}
```
2. Yifan Chen, Houman Owhadi, Andrew M. Stuart. "[Consistency of Empirical Bayes And Kernel Flow For Hierarchical Parameter Estimation](https://arxiv.org/abs/2005.11375), Mathematics of Computation, 2021.
```
@article{chen2021consistency,
  title={Consistency of empirical Bayes and kernel flow for hierarchical parameter estimation},
  author={Chen, Yifan and Owhadi, Houman and Stuart, Andrew},
  journal={Mathematics of Computation},
  volume={90},
  number={332},
  pages={2527--2578},
  year={2021}
}
```
3. Yifan Chen, Houman Owhadi, Florian Schaefer. "[Sparse Cholesky Factorization for Solving Nonlinear PDEs via Gaussian Processes](https://arxiv.org/abs/2304.01294)", Mathematics of Computation, 2024.
```
@article{chen2025sparse,
  title={Sparse Cholesky factorization for solving nonlinear PDEs via Gaussian processes},
  author={Chen, Yifan and Owhadi, Houman and Sch{\"a}fer, Florian},
  journal={Mathematics of Computation},
  volume={94},
  number={353},
  pages={1235--1280},
  year={2025}
}
```
4. Pau Batlle, Yifan Chen, Bamdad Hosseini, Houman Owhadi, Andrew M. Stuart. "[Error Analysis of Kernel/GP Methods for Nonlinear and Parametric PDEs](https://arxiv.org/abs/2305.04962)", Jounal of Computational Physics, 2024.
```
@article{batlle2025error,
  title={Error analysis of kernel/GP methods for nonlinear and parametric PDEs},
  author={Batlle, Pau and Chen, Yifan and Hosseini, Bamdad and Owhadi, Houman and Stuart, Andrew M},
  journal={Journal of Computational Physics},
  volume={520},
  pages={113488},
  year={2025},
  publisher={Elsevier}
}
```
5. Yifan Chen, Bamdad Hosseini, Houman Owhadi, Andrew M Stuart. "[Gaussian Measures Conditioned on Nonlinear Observations: Consistency, MAP Estimators, and Simulation](https://arxiv.org/abs/2405.13149)", Statistics and Computing, 2024.
```
@article{chen2025gaussian,
  title={Gaussian measures conditioned on nonlinear observations: consistency, MAP estimators, and simulation},
  author={Chen, Yifan and Hosseini, Bamdad and Owhadi, Houman and Stuart, Andrew M},
  journal={Statistics and Computing},
  volume={35},
  number={1},
  pages={10},
  year={2025},
  publisher={Springer}
}
```
Some of the analytic results for multiscale PDEs (especially the sparse Cholesky fast solver) are also related to the following works (with [corresponding repository](https://github.com/yifanc96/Multiscale-via-Subsampling/tree/master))

5. Yifan Chen, Thomas Y. Hou. "[Multiscale Elliptic PDEs Upscaling and Function Approximation via Subsampled Data](https://arxiv.org/abs/2010.04199)", SIAM Multiscale Modeling and Simulation, 2022.

6. Yifan Chen, Thomas Y. Hou. "[Function Approximation via The Subsampled Poincare Inequality](https://arxiv.org/abs/1912.08173)", Discrete & Continuous Dynamical Systems - A, 2020.


