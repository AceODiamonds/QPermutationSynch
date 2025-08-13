# Quantum Permutation Synchronization

This repository is created to host the final project we're developing for our "Quantum Computing" course.

## Project Overview

Permutation synchronization is the problem of recovering consistent feature point matches across multiple views or images by ensuring cycle consistency of pairwise permutations. This project implements and evaluates several approaches:

- Classical Simulated Annealing  
- Quantum Approximate Optimization Algorithm (QAOA)  

The problem is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem which can be solved using either classical optimization techniques or quantum methods.


Willow Images → Keypoints → AlexNet features → Similarity matrix → Hungarian algorithm → Noisy P_ij 
→ QUBO formulation → Recovered X_i (permutations) → Evaluation

## File Structure

### ● Main Implementation Files:
- `annealing.py` – Classical simulated annealing implementation  
- `qaoa_final.py` – QAOA and VQE implementation using Qiskit  
- `syn_data_processing.py` – Synthetic data generation and processing utilities  
- `main.py` – Entry point for running experiments(not used at the moment since we're testing solely on Jupyter notebooks)  

### ● Jupyter Notebooks:
- `annealing.ipynb`, `annealing2.ipynb` – Interactive versions of simulated annealing  
- `qiskit_ws_qaoa_final.ipynb` – QAOA with Qiskit workspace  
- `qsychn_final.ipynb`, `qsychn_qaoa_final.ipynb` – Final notebook implementations  

### ● Utility Directories:
- `utils/` – Helper functions for data processing and methods  
- `implementations/` – Different implementation approaches(legacy, simple and minimal implementations)  
- `PF-dataset/` – Willow dataset containing images with feature points  

### ● Results and Data:
- `results/` – Output results and visualizations  
- `synthetic_data/` – Generated synthetic datasets  
- `synth_data_results/` – Results from synthetic data experiments  
- `synchronized_images_simulatedAnnealing/` – Visualization of synchronization results  
