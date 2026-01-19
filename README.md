# Hybrid-Models-for-parameter-description
This repository contains a set of Python scripts that I developed to model friction force and injection dynamics in spring-driven autoinjectors using an hybrid modellying framework. The approach combines a mechanistic friction model with a data-driven contribution based on  Artificial Neural Networks (ANNs) and propagates the epistemic uncertainty from the ANN through to the predicted injection time via Monte Carlo (MC) simulations.

The codes are intended for research and engineering (process system and healthcare engineering) focused on model calibration, uncertinty propagation and performance analysis of autoinjector systems.

1. **Hybrid Friction Force Model**\
   Purpose: this script evaluates the friction force acting on the plunger by combining:
    - a mechanistic friction model derived from viscous shear considerations, and
    - a data-driven ANN trained to predict a calibration parameter as a function of system state
2. **Injection Time Evaluation**\
   Purpose: this script computes the **injection time** of a spring-driven autoinjector by solving the plunger dynamics using the hybrid friction model obtained in the previous step.
3. **Monte Carlo Simulation and DoE**\
   Purpose: this script propagates **epistemic uncertainty** arising from the ANN through the hybrid model to quantify uncertainty in the predicted injection time.
   


