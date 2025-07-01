# R-MDN
A Recurrent Mixture Density Network implementation for the analysis of experimental rodent decision making data (Restaurant Row).

## Authors:
- Cameron Jordan (UC Berkeley)

**The architecture has changed several times**
Within each weights folder, find the corresponding architecture; which can be used to load the weights. 

There are two primary architectures:
1. Recurrent Mixture Density Network, which employs a two Gaussian mixture; with auxiliary loss functions to align the different modes to the Accept/Reject modes.
2. Recurrent Gaussian Network, which employs a single Gaussian; where the Mu parameter is restricted to [0, +), and Phi acts as a weighting parameter - hopefully aligning with the decision of the network (weights the Gaussian towards the negative under Reject regimes, and otherwise keeps it in the positive domain) -- ideally, the Phi parameter would also control the speed of the rise/width of the bump.

## TODO:
- [ ] Evaluate/test manifold visualization functions
    - [ ] Update manifold visualizations to include UMAP (Uniform Manifold Approximation and Projection)
- [ ] Evaluate manifold activity under various training paradigms: 
    1. Curriculum learning (the experimental paradigm under which the rodents are trained)
    2. General training paradigm (no separation of training data/trials by reward probability)
- [ ] Build statistical tools to evaluate the model performance relative to the experimental data distribution

## License:
MIT License (2025)