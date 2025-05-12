'''
This module provides the architecture and loss functions that characterize 
recurrent mixture density networks

Methods
-------
    RMDN : class

    mdn_loss
        Parameters
        ----------
            pi : pytorch.Tensor
            mu : pytorch.Tensor
            sigma : pytorch.Tensor
            targets : pytorch.Tensor
            lambda_s : float
            eps : float
'''

import torch
import torch.nn as nn
import numpy as np

#######################################################
## Mixture Density Network Class                     ##
#######################################################

class RMDN(nn.Module):
    '''
    Attributes
    ----------
        hidden_size : int
        input_size : int
        output_size : int
        num_gaussians : int
        feedback_size : int (= output_size * num_gaussians)
        rnn : nn.GRU
        fc : nn.Linear
        pi : nn.Linear
        mu : nn.Linear
        sigma : nn.Linear
    
    Methods
    -------
        __init__(hidden_size, input_size, output_size, num_gaussians)
        forward(x, epoch, max_epochs, train, outputs, return_hidden)
    
    '''
    def __init__(self, hidden_size, input_size=2, output_size=1, num_gaussians=3):
        '''
        
        '''
        super(RMDN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_gaussians = num_gaussians
        self.feedback_size = self.output_size * self.num_gaussians
        
        # Recurrent Layers
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        # fc Layer
        # fc in_features: hidden_size + mus_{t-1} + sigmas_{t-1} + y_{t-1}
        self.fc = nn.Linear(in_features=(self.hidden_size + (2 * self.feedback_size) + self.output_size), out_features=(2 * self.hidden_size))

        # Mixture Density Output Layers
        self.pi = nn.Linear(in_features=(2 * self.hidden_size), out_features=self.num_gaussians) # Mixture cofficients
        self.mu = nn.Linear(in_features=(2 * self.hidden_size), out_features=self.feedback_size) # Means
        self.sigma = nn.Linear(in_features=(2 * self.hidden_size), out_features=self.feedback_size) # Variances

        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''

        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)

    def forward(self, x, epoch=None, max_epochs=None, train=True, outputs=None, return_hidden=False):
        '''
        Run the forward pass through the network

        Parameters
        ----------
            x: Tensor of shape (batch_size, seq_len, input_size)
            epoch : int
            max_epochs : int
            train : bool, optional
            outputs: Tensor of shape (batch_size, seq_len, output_size) used for teacher forcing.
            verbose : bool, optional

        Returns
        -------
            pis : 
            mus :
            sigmas :
            output :
            h_states :
        
        Notes
        -----
            feedback: during training, y_{t-1} is taken from outputs (teacher forcing)
                      during inference, it is sampled from the output distribution
                    
            sigma: represents standard deviations (not variances).
        '''

        batch_size, seq_len, _ = x.size()
        mu_t_minus_one = torch.zeros(batch_size, 1, self.feedback_size, device=x.device)
        sigma_t_minus_one = torch.zeros(batch_size, 1, self.feedback_size, device=x.device)
        y_t_minus_one = torch.zeros(batch_size, 1, self.output_size, device=x.device)

        if train:
            assert outputs is not None, 'Training requires non None-type outputs parameter'
            assert epoch is not None, 'Training requires non None-type epoch parameter'
            assert max_epochs is not None, 'Training requires non None-type max_epochs parameter'
        else:
            output = []
            output.append(y_t_minus_one)

        pis, mus, sigmas = [], [], []
        h_states = []

        for t in range(seq_len):
            # Recurrent Layer Output
            x_t = x[:, t, :].unsqueeze(1)
            h_out, _ = self.rnn(x_t)
            h_states.append(h_out)

            # fc Layer Output
            # append previous mus, sigmas and output_value to the fc layer that feeds forward to mu and sigma
            h_conditioned = torch.cat((h_out, mu_t_minus_one, sigma_t_minus_one, y_t_minus_one), dim=-1)
            h_stacked = torch.tanh(self.fc(h_conditioned))

            # Mixture Density Outputs
            pi_t = self.pi(h_stacked).view(h_stacked.size(0), h_stacked.size(1), self.num_gaussians)
            pi_t = nn.functional.softmax(pi_t, dim=-1)
            pis.append(pi_t)

            mu_t = self.mu(h_stacked).view(h_stacked.size(0), h_stacked.size(1), self.num_gaussians, self.output_size)
            mus.append(mu_t)
            
            sigma_t = self.sigma(h_stacked).view(h_stacked.size(0), h_stacked.size(1), self.num_gaussians, self.output_size)
            sigma_t = torch.clamp(nn.Softplus()(sigma_t), min=1e-3)
            sigmas.append(sigma_t)

            mu_t_minus_one, sigma_t_minus_one = mu_t.flatten(start_dim=2), sigma_t.flatten(start_dim=2)

            if train:
                ground_truth = outputs[:, t, :].unsqueeze(1)
                sampled_output = self._sample_output(pi_t, mu_t, sigma_t)

                # Stochastic Teacher/Target Forcing Policy
                # Bengio, S. et al. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NIPS.
                # Teacher Forcing for first n/2 epochs before introducing scheduled sampling
                if epoch < 0.5 * max_epochs:
                    y_t_minus_one = ground_truth
                elif np.random.rand() < (epoch - 0.5 * max_epochs / 0.5 * max_epochs):
                    y_t_minus_one = sampled_output
                else:
                    y_t_minus_one = ground_truth

            else:
                y_t_minus_one = self._sample_output(pi_t, mu_t, sigma_t)
                output.append(y_t_minus_one)

        pis = torch.stack(pis, dim=1).squeeze(2)
        mus = torch.stack(mus, dim=1).squeeze(2)
        sigmas = torch.stack(sigmas, dim=1).squeeze(2)
        h_states = torch.stack(h_states, dim=1)
        
        if train:
            return pis, mus, sigmas, h_states
        else:
            output = torch.stack(output, dim=1).squeeze(2)
            if return_hidden:
                return pis, mus, sigmas, output, h_states
            else:
                return pis, mus, sigmas, output

    def _sample_output(self, pi, mu, sigma):
        '''
        Sample one output per batch (for the current time step) from the mixture ~ N(mu_cond, sigma_cond).
            
            Assumes:
            pi:  (batch, 1, num_gaussians)
            mu:  (batch, 1, num_gaussians, output_size)
            sigma: (batch, 1, num_gaussians, output_size)
            
            Returns:
            A sample of shape (batch, 1, output_size)
        '''

        pi = pi.squeeze(1)
        mu = mu.squeeze(1)
        sigma = sigma.squeeze(1)
        
        # Sample a mixture component for each batch element.
        component_indices = torch.multinomial(pi, num_samples=1)  # (batch, 1)
        component_indices_expanded = component_indices.unsqueeze(-1).expand(-1, -1, mu.size(-1))  # (batch, 1, output_size)
        chosen_mu = torch.gather(mu, 1, component_indices_expanded).squeeze(1)       # (batch, output_size)
        chosen_sigma = torch.gather(sigma, 1, component_indices_expanded).squeeze(1) # (batch, output_size)
        
        noise = torch.randn_like(chosen_mu)
        sample = chosen_mu + chosen_sigma * noise
        return sample.unsqueeze(1)  # shape: (batch, 1, output_size)

#######################################################
## Mixture Density Network Loss                      ##
## (Implicitely) Conditional Negative Log-Likelihood ##
#######################################################

def mdn_loss(pi, mu, sigma, targets, lambda_s=0.01, eps=1e-8):
    '''
    Computes the conditional negative log-likelihood for the MDN output.

    Parameters
    ----------
        pi      : Tensor of shape (batch, seq_len, num_gaussians)
        mu      : Tensor of shape (batch, seq_len, num_gaussians, output_size)
        sigma   : Tensor of shape (batch, seq_len, num_gaussians, output_size) (assumed to be std)
        targets : Tensor of shape (batch, seq_len, output_size)
        
    Returns
    -------
        loss: Tensor of scalars
    '''

    # Add Entropy Regularization to the Mixture Weights
    # => Entropy loss prevents extreme confidence in one mixture component, hopefully leading to smoother output dynamics
    entropy_loss = -lambda_s * torch.sum(pi * torch.log(pi + eps), dim=-1)
    entropy_loss = entropy_loss.mean()
    
    targets = targets.unsqueeze(2).expand_as(mu)  # Match shape for mixture components
    pi = pi.unsqueeze(-1).expand_as(mu) # Reshape pi to use the same mixing coefficients for all outputs

    dist = torch.distributions.Normal(mu, sigma)
    likelihood = torch.exp(dist.log_prob(targets))  # Probability density
    weighted_likelihood = torch.sum(pi * likelihood, dim=-1)
    loss = -torch.log(weighted_likelihood + eps).mean()  # Negative log-likelihood

    return loss + entropy_loss
