'''

Methods
-------
    RMDN : class

    mdn_loss
        Parameters
        ----------
            phi : pytorch.Tensor
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
## Recurrent Gaussian Network Class                  ##
#######################################################

class RGN(nn.Module):
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
                 output_size=1):
        '''

        '''
        super(RGN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.feedback_size = 1

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
        
        self.phi = nn.Linear(in_features=2 * self.hidden_size,
                             out_features=1)
        self.mu = nn.Linear(in_features=2 * self.hidden_size,
                             out_features=1)
        self.sigma = nn.Linear(in_features=2 * self.hidden_size,
                             out_features=1)

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
        y_t_minus_one = torch.zeros(batch_size, 1, self.output_size, device=x.device)

        if train:
            assert epoch is not None, 'Training requires non None-type epoch parameter'
            assert max_epochs is not None, 'Training requires non None-type max_epochs parameter'
        
        output = [y_t_minus_one]
        phis, mus, sigmas = [], [], []
        h_states = []
        hidden, output_hidden = None, None

        for t in range(seq_len):
            # Recurrent Layer Output
            x_t = x[:, t, :].unsqueeze(1)

            # Feedback previous output
            x_stacked_t = torch.cat((x_t, y_t_minus_one), dim=-1)

            # Get RNN output
            h_rnn, hidden = self.rnn(x_stacked_t, hidden)
            h_states.append(h_rnn)

            # Process through FC layer
            h_stacked = torch.relu(self.fc(h_rnn))

            # Gaussian parameters
            phi_t = self.phi(h_stacked)
            phi_t = nn.Tanh()(phi_t) # restrict to [-1, 1]
            mu_t = self.mu(h_stacked)
            mu_t = nn.Softplus()(mu_t) # restrict to [0, +]
            sigma_t = self.sigma(h_stacked)
            sigma_t = torch.clamp(nn.Softplus()(sigma_t), min=1e-3)

            phis.append(phi_t)
            mus.append(mu_t)
            sigmas.append(sigma_t)

            if train:
                ground_truth = outputs[:, t, :].unsqueeze(1)

                sampled_output = self._sample_output(phi_t, mu_t, sigma_t)

                # Process through Output RNN
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
                sampled_output = self._sample_output(phi_t, mu_t, sigma_t)

                # Process through Output RNN
                concatenated_flat = torch.cat((sampled_output, y_t_minus_one), dim=-1)
                gru_out, output_hidden = self.output_rnn(concatenated_flat, output_hidden)
                true_output = self.output_linear(gru_out)
                output.append(true_output)
                
        # Stack outputs
        phis = torch.stack(phis, dim=1).squeeze(2)
        mus = torch.stack(mus, dim=1).squeeze(2)
        sigmas = torch.stack(sigmas, dim=1).squeeze(2)
        output = torch.stack(output, dim=1).squeeze(2)
        h_states = torch.stack(h_states, dim=1)

        if train:
            return phis, mus, sigmas, output
        else:
            if return_hidden:
                return phis, mus, sigmas, output, h_states
            else:
                return phis, mus, sigmas, output

    def _sample_output(self,
                       phi,
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

        phi, mu, sigma = phi.squeeze(1), mu.squeeze(1), sigma.squeeze(1)

        noise = torch.randn_like(mu)
        sample = phi * mu + sigma * noise
        return sample.unsqueeze(1)  # shape: (batch, 1, output_size)

#######################################################
## Mixture Density Network Loss                      ##
## (Implicitly) Conditional Negative Log-Likelihood  ##
#######################################################

def nll_loss(phi, mu, sigma, targets):
    '''
    Computes the conditional negative log-likelihood for the MDN output.

    Parameters
    ----------
        pi      : Tensor of shape (batch, seq_len, num_gaussians)
        mu      : Tensor of shape (batch, seq_len, num_gaussians, output_size)
        sigma   : Tensor of shape (batch, seq_len, num_gaussians, output_size) (assumed to be std)
        targets : Tensor of shape (batch, seq_len, output_size)
        prob    : Tensor of shape (batch, seq_len)

    Returns
    -------
        loss: Tensor of scalars
    '''

    # Negative log likelihood
    dist = torch.distributions.Normal(phi * mu, sigma)
    nll_loss = -dist.log_prob(targets).mean()
    #likelihood = torch.exp(dist.log_prob(targets))  # Probability density
    #nll_loss = -torch.log(likelihood + eps).mean()  # Negative log-likelihood

    # Penalize excessive variance
    variance_penalty = 0.1 * torch.mean(sigma**2)

    return nll_loss + variance_penalty
