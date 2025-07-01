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
    def __init__(self,
                 hidden_size,
                 input_size=2,
                 output_size=1,
                 num_gaussians=2):
        '''

        '''
        super(RMDN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_gaussians = num_gaussians
        self.feedback_size = self.num_gaussians + 2 * (self.num_gaussians * self.output_size)

        # Recurrent Layer
        # self.rnn learns to parameterize the distribution
        self.rnn = nn.GRU(input_size=(self.input_size + self.feedback_size),
                          hidden_size=self.hidden_size,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        
        # Integration Layer
        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=2 * self.hidden_size)

        # Mixture Density Output Layers
        self.pi = nn.Linear(in_features=2 * self.hidden_size,
                            out_features=self.num_gaussians) # Mixture cofficients
        self.mu = nn.Linear(in_features=2 * self.hidden_size,
                            out_features=self.num_gaussians) # Means
        self.sigma = nn.Linear(in_features=2 * self.hidden_size,
                               out_features=self.num_gaussians) # Variances

        # Process/Smooth Output
        # self.output learns to produce cohesive 'bump-like' output structures
        self.output_rnn = nn.GRU(input_size=2 * self.output_size,
                   hidden_size=self.hidden_size,
                   batch_first=True)
        self.output_linear = nn.Linear(in_features=self.hidden_size,
                      out_features=self.output_size)

        self.apply(self._init_weights)

    def _init_weights(self,
                      module):
        '''
        Initialize weights
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)

    def forward(self,
                x,
                epoch=None,
                max_epochs=None,
                train=True,
                outputs=None,
                return_hidden=False):
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

        # Initialize previous timestep values
        pi_t_minus_one = torch.zeros(batch_size, 1, self.num_gaussians, device=x.device)
        mu_t_minus_one = torch.zeros(batch_size, 1, self.num_gaussians * self.output_size, device=x.device)
        sigma_t_minus_one = torch.zeros(batch_size, 1, self.num_gaussians * self.output_size, device=x.device)
        y_t_minus_one = torch.zeros(batch_size, 1, self.output_size, device=x.device)

        if train:
            assert outputs is not None, 'Training requires non None-type outputs parameter'
            assert epoch is not None, 'Training requires non None-type epoch parameter'
            assert max_epochs is not None, 'Training requires non None-type max_epochs parameter'
        
        output = []
        output.append(y_t_minus_one)

        pis, mus, sigmas = [], [], []
        h_states = []

        hidden, output_hidden = None, None

        for t in range(seq_len):
            # Recurrent Layer Output
            x_t = x[:, t, :].unsqueeze(1)

            feedback_t = torch.cat((
                pi_t_minus_one,
                mu_t_minus_one,
                sigma_t_minus_one
            ), dim=-1)

            x_stacked_t = torch.cat((x_t, feedback_t), dim=-1)

            h_rnn, hidden = self.rnn(x_stacked_t, hidden)
            h_states.append(h_rnn)

            # fc Layer Output
            h_stacked = torch.relu(self.fc(h_rnn))

            # Mixture Density Outputs
            pi_t = self.pi(h_stacked).view(batch_size, 1, self.num_gaussians)
            pi_t = nn.functional.softmax(pi_t, dim=-1)
            pis.append(pi_t)

            mu_t = self.mu(h_stacked).view(batch_size, 1, self.num_gaussians, self.output_size)
            mus.append(mu_t)

            sigma_t = self.sigma(h_stacked).view(batch_size, 1, self.num_gaussians, self.output_size)
            sigma_t = torch.clamp(nn.Softplus()(sigma_t), min=1e-3)
            sigmas.append(sigma_t)

            # Prepare for next timestep
            pi_t_minus_one = pi_t
            mu_t_minus_one =  mu_t.flatten(start_dim=2)
            sigma_t_minus_one = sigma_t.flatten(start_dim=2)

            if train:
                ground_truth = outputs[:, t, :].unsqueeze(1)
                sampled_output = self._sample_output(pi_t, mu_t, sigma_t)
                concatenated_flat = torch.cat((sampled_output, y_t_minus_one), dim=-1)
                gru_out, output_hidden = self.output_rnn(concatenated_flat, output_hidden)
                true_output = self.output_linear(gru_out)
                output.append(true_output)

                # Stochastic Teacher/Target Forcing Policy
                # Bengio, S. et al. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NIPS.
                
                # Scheduled Sampling
                teacher_forcing_prob = max(0.0, 1.0 - (epoch / max_epochs * 0.7)) # Transition faster to model outputs
                if np.random.rand() < teacher_forcing_prob:
                    y_t_minus_one = ground_truth
                else:
                    y_t_minus_one = true_output

            else:
                sampled_output = self._sample_output(pi_t, mu_t, sigma_t)
                concatenated_flat = torch.cat((sampled_output, y_t_minus_one), dim=-1)
                gru_out, output_hidden = self.output_rnn(concatenated_flat, output_hidden)
                true_output = self.output_linear(gru_out)
                output.append(true_output)
                
        # Stack outputs
        pis = torch.stack(pis, dim=1).squeeze(2)
        mus = torch.stack(mus, dim=1).squeeze(2)
        sigmas = torch.stack(sigmas, dim=1).squeeze(2)
        h_states = torch.stack(h_states, dim=1)

        if train:
            output = torch.stack(output, dim=1).squeeze(2)
            return pis, mus, sigmas, output
        else:
            output = torch.stack(output, dim=1).squeeze(2)
            if return_hidden:
                return pis, mus, sigmas, output, h_states
            else:
                return pis, mus, sigmas, output

    def _sample_output(self,
                       pi,
                       mu,
                       sigma):
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
## (Implicitly) Conditional Negative Log-Likelihood  ##
#######################################################

def mdn_loss(pi,
             mu,
             sigma,
             output,
             targets,
             prob,
             lambda_s=0.1,
             lambda_log=0.5,
             eps=1e-8):
    '''
    Computes the conditional negative log-likelihood for the MDN output.

    Parameters
    ----------
        pi      : Tensor of shape (batch, seq_len, num_gaussians)
        mu      : Tensor of shape (batch, seq_len, num_gaussians, output_size)
        sigma   : Tensor of shape (batch, seq_len, num_gaussians, output_size) (assumed to be std)
        output  : Tensor of shape (batch, seq_len, output_size)
        targets : Tensor of shape (batch, seq_len, output_size)
        prob    : Tensor of shape (batch, seq_len)

    Returns
    -------
        loss: Tensor of scalars
    '''

    # Add Entropy Regularization to the Mixture Weights
    ## Entropy loss prevents extreme confidence in one mixture component, 
    ## hopefully leading to smoother output dynamics
    entropy_loss = -lambda_s * torch.sum(pi * torch.log(pi + eps), dim=-1)
    entropy_loss = entropy_loss.mean()


    conditioning_loss = lambda_log * torch.nn.functional.cross_entropy(
        pi.view(-1, 2), # (batch*seq_len, 2)
        prob.view(-1).long() #(batch*seq_len,) - convert to class indices
    )

    # Negative log likelihood
    targets = targets.unsqueeze(2).expand_as(mu)  # Match shape for mixture components
    pi = pi.unsqueeze(-1).expand_as(mu) # Reshape pi

    dist = torch.distributions.Normal(mu, sigma)
    likelihood = torch.exp(dist.log_prob(targets))  # Probability density
    weighted_likelihood = torch.sum(pi * likelihood, dim=2)
    nll_loss = -torch.log(weighted_likelihood + eps).mean()  # Negative log-likelihood

    # Penalize excessive variance
    variance_penalty = 0.1 * torch.mean(sigma**2)

    return nll_loss + entropy_loss + conditioning_loss + variance_penalty
