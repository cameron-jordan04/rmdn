# R-MDN
A Recurrent Mixture Density Network implementation for the analysis of experimental rodent decision making data (Restaurant Row).

## Authors:
- Cameron Jordan (UC Berkeley)

## TODO:
- [ ] Rigorously test dataloader
- [ ] Evaluate the effects of the ramp vs. sinusoidal positional encoding on model performance
- [ ] Evaluate manifold activity under various training paradigms: 
    1. Curriculum learning (the experimental paradigm under which the rodents are trained)
    2. General training paradigm (no separation of training data/trials by reward probability)
- [ ] Build statistical tools to evaluate the model performance relative to the experimental data distribution