# Investigating the Loss Landscape of Deep Neural Networks

**Course:** Neural Processing & Systems (TEC-323)  
**Term:** 2026  
**Author(s):** [Add names]  
**Institution:** [Add institution]  
**Submission date:** [Add date]

> This is a report scaffold. Replace every bracketed prompt with evidence from
> actual runs. Do not invent results. Interpret landscape figures as slices,
> not proof of minima or saddle points.

## Abstract

[Summarize the research question, Two Moons dataset, experimental factors,
methods, measured results, and principal evidence-based conclusion.]

## 1. Introduction

[Introduce neural-network optimization and why architecture, optimizer, and
initialization may affect training and observed loss geometry.]

## 2. Problem Statement and Objectives

[State the problem and list the objectives addressed by the experiments.]

## 3. Theoretical Background

### 3.1 Feedforward neural networks
[Explain layers, weights, biases, activations, and logits.]

### 3.2 Classification loss
[Explain cross-entropy and why the model returns logits.]

### 3.3 Gradient descent and training
[Explain forward propagation, backpropagation, epochs, and learning rate.]

### 3.4 SGD, Momentum, and Adam
[Describe the update methods and the hyperparameters used.]

### 3.5 Weight initialization
[Describe PyTorch default, Xavier uniform, and He/Kaiming uniform.]

### 3.6 Loss landscapes
[Define the loss as a function of parameters; distinguish a 2D slice from the
full high-dimensional landscape.]

### 3.7 Minima, saddle points, flatness, and sharpness
[Define the terms and explain why a 2D plot alone cannot establish them in the
full parameter space.]

## 4. Dataset and Preprocessing

[Describe the 2,000-sample Two Moons dataset, noise, fixed random state,
stratified 80/20 split, training-only standardization, and tensor formats.]

## 5. Methodology and Experimental Setup

[Record OS, Python, package versions, hardware, seeds, epochs, learning rate,
loss, evaluation procedure, and execution date. Link the GitHub repository:
[add URL].]

## 6. Network Architectures

[Describe input/output dimensions, hidden widths, depths, and ReLU activations.]

## 7. Optimizer Comparison

[Compare SGD, Momentum (0.9), and Adam while keeping the architecture,
initialization, split, epochs, and learning rate fixed. Insert the actual
comparison table and training curves.]

## 8. Architecture Comparison

[Discuss depth and width comparisons separately. State which variables were
held fixed.]

## 9. Initialization Comparison

[Compare Xavier and He with the same architecture, optimizer, data, and
training setup.]

## 10. Loss Landscape Visualization

[Explain checkpoint, training-loss surface, coordinate range, resolution, and
direction-generation method. Discuss only the sampled 2D slice.]

**Figure 1.** [Insert `two_moons_dataset.png` and a descriptive caption.]

**Figure 2.** [Insert a training-loss plot and explain axes and observations.]

**Figure 3.** [Insert the 3D loss-slice plot and describe the observed slice.]

**Figure 4.** [Insert the contour plot; identify the trained point.]

## 11. Results and Discussion

[Insert measured tables/figures. Report multiple-seed means and standard
deviations where applicable. Explain variation and avoid causal claims beyond
the controlled comparisons.]

## 12. Limitations

- Two Moons is a small synthetic dataset, not a large-scale real-world task.
- Results depend on the selected architectures, hyperparameters, seeds, and
  finite training budget.
- A two-dimensional slice does not represent the complete parameter space.
- The displayed slice alone cannot prove a local/global minimum or saddle
  point.
- [Add observed limitations, including compute or reproducibility constraints.]

## 13. Conclusion and Future Work

[Summarize evidence-based findings and propose additional seeds, width/depth
settings, visualization directions, curvature analysis, or real datasets.]

## 14. References

[Add credible sources actually consulted, using the citation format required
by the course.]

## Appendix A: Source Code and Execution Instructions

[Link the source repository and include the commands used. List package
versions from `requirements-lock.txt` and explain how to regenerate the
figures and tables.]
