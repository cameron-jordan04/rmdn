'''
This module encapsulates the dataset used to train the recurrent mixture density network
on both synthetic and experimental data

Methods
------
    DecisionDataset : class
'''

# Imports

import math
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

# Basic logging configuration (logs INFO and above to console)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default data generation parameters
DEFAULT_PARAMS = {
        'sample_rate' : 60,                  # Samples per second
        'seq_len' : 5 * 60,                  # Sequence length in samples (5 seconds)
        'mean_duration' : 0.25,              # Mean duration for synthetic decision time (seconds)
        'std_dev' : 0.25,                    # Default std dev for synthetic Gaussian bump (seconds)
        'noise_std' : 0.05,                  # Standard deviation for added noise
        'magnitude' : 1.0,                   # Peak magnitude of the target bump
        'min_experimental_std_dev' : 0.05,   # Min std dev for experimental bumps (seconds)
}

# Experimental data df columns used by experimental dataloader
REQUIRED_COLUMNS = [
    'animal', 'session', 'trial', 'offer_prob', 'slp_accept', 'quit', 'slp_tone_onset_time', 
    'slp_T_Entry_time', 'slp_choice_time', 'reward', 'restaurant'
]

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device: %s.', device)

#######################################################
## Input Dataset Functions                           ##
#######################################################

class DecisionDataset(Dataset):
    '''
    PyTorch Dataset for either:
        1. Generating synthetic training data
        2. Loading and processing experimental data from specified sessions, or from all sessions

    Attributes
    ----------
        generate : bool
            Mode indicator, (True: generate, False: load)
        num_samples : int
            Number of samples in the dataset
        params : dict
            Parameters used for data generation or processing
        time : np.ndarray
            Time vector for the sequence length
        experimental_trial_params : list[dict]
            Stores processed parameters for each experimental trial when generate=False

    Methods
    -------
        __init__(generate, num_samples, params, add_noise, df)
            Usage
            -----
                * In order to load all sessions: generate=False, df must be provided
                * In order to generate synthetic data: generate=True
                if num_samples is provided, then num_samples will be generated (default = 10000)

        __getitem__(index, p)
            Usage
            ----
                * If generate=False, then index is used
                * If generate=True, then index is ignored, and p is used if provided

        plot_trial(index, p)
            Usage
            -----
                * If generate=False, then index is used
                * If generate=True, then index is ignored, and p is used if provided
    '''

    def __init__(self,
                 generate : bool,
                 num_samples=None,
                 params=None,
                 add_noise=False,
                 df=None):
        '''
        Initializes the Dataset class

        Parameters
        ----------
            generate : bool
                If True, generate synthetic data, if False, load experimental data
            num_samples : int, optional
                The number of samples to virtually expose in the dataset, used by __len__
                Ignored if generate=False, Defaults to 10000 if num_samples=None
            params : dictionary, optional
                The dictionary to override default parameters (DEFAULT_DICT)
            add_noise : bool
                If True, add Gaussian noise to inputs and targets
            df : pandas DataFrame
                Ignored if generate=False
        '''

        super().__init__()

        self.generate = generate
        self.add_noise = add_noise

        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        for key in ['seq_len', 'sample_rate', 'noise_std', 'magnitude']:
            if key not in self.params:
                raise ValueError(f"Missing essential parameter '{key}' in parameter dictionary")

        self.time = np.arange(self.params['seq_len']) / self.params['sample_rate']
        self.df = df

        if self.generate:
            # --- Generation Mode ---
            self.num_samples = num_samples if num_samples is not None else 10000
            if 'mean_duration' not in self.params or 'std_dev' not in self.params:
                raise ValueError('Missing mean_duration or std_dev in params for generation mode.')

            logger.info("Initialized Dataset in GENERATE mode with %s samples.", self.num_samples)
        else:
            # --- Experimental Data Loading Mode ---
            logger.info('Initialized Dataset in LOAD mode.')
            self.num_samples = 0

            try:
                self._preprocess_experimental_data()
                self.num_samples = len(self.df)

                if self.num_samples == 0:
                    logger.warning('Data loading finished, but no valid trials were processed.')
                else:
                    logger.info('Successfully loaded and processed %s experimental trials.', self.num_samples)

            except (FileNotFoundError, ValueError, KeyError) as e:
                logger.error('Failed to initialize dataset in LOAD mode: %s', e)
                raise

    def __len__(self):
        ''' Returns the total number of samples in the dataset '''
        return self.num_samples

    def _build_sample_tensors(self,
                              p,
                              decision,
                              bump_peak_time_idx,
                              std_dev_samples,
                              zero_until_idx):
        '''
        Builds input and target tensors for a single sample given trial parameters
            This logic is reused for both generated and experimental data 
            (using their resp. parameters)

        Parameters
        ----------
            p : float
                Probability value for the input feature
            decision : int
                Decision value (1 for accept, 0 for reject)
            bump_peak_time_idx : int
                Index on the sequence time axis where the synthetic target bump peak should occur
            std_dev_samples : float
                The standard deviation of the Gaussian bump
            zero_until_idx : int
                Index on the sequence time axis before which the synthetic data should be zero
            add_noise : bool, optional
                If True, adds noise based on self.params['noise_std']

        Returns
        -------
            inputs_tensor : torch.Tensor
                shape : (self.params['seq_len'], input_features)
            target_tensor : torch.Tensor
                shape : (self.params['seq_len'],)
        '''

        # --- Create Input Features ---
        # Feature 1 - the probabilities are zero meaned for increased interpretability
        prob_input = np.full(self.params['seq_len'], p - 0.5)

        # Feature 2
        # positional_encoding = 0.25 * np.sin(2 * np.pi * self.time)

        # Feature 3
        ramp = self.time / self.time[-1] if self.params['seq_len'] > 1 and self.time[-1] > 0 else np.zeros_like(self.time)

        # Stack features
        # inputs_np = np.stack((prob_input, positional_encoding, ramp), axis=1)
        inputs_np = np.stack((prob_input, ramp), axis=1)
        inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32)

        # --- Create Target Features ---
        target_np = np.zeros(self.params['seq_len'])
        time_indices = np.arange(self.params['seq_len'])
        exponent = -0.5 * ((time_indices - bump_peak_time_idx) / max(std_dev_samples, 1e-6)) ** 2
        bump_shape_np = np.exp(exponent)

        # Scale bump to desired magnitude
        target_np = bump_shape_np * self.params['magnitude']

        if decision <= 0:
            target_np *= -1.0

        if zero_until_idx > 0:
            target_np[:zero_until_idx] = 0.0

        target_tensor = torch.tensor(target_np, dtype=torch.float32)

        # --- Add noise (Optional) ---
        if self.add_noise and self.params['noise_std'] > 0:
            inputs_tensor += torch.randn_like(inputs_tensor) * self.params['noise_std']
            target_tensor += torch.randn_like(target_tensor) * self.params['noise_std']

        return inputs_tensor, target_tensor

    def _preprocess_experimental_data(self):
        '''
        Processes the combined dataframe by appending several columns containing 
        parameters for _build_sample_tensors
            Note: The tensor building happens on demand in __getitem__
        '''

        peak_times, std_devs, zero_until  = [], [], []
        drop_idxs = []

        for index, row in self.df.iterrows():

            # Track and skip quitting rows
            if not math.isnan(row['quit']):
                drop_idxs.append(index)
                continue
            else:
                trial_key = str(row['animal']) + '_' + str(row['session']) + '_' + str(row['trial'])

                tone_onset_sec = float(row['slp_tone_onset_time'])
                t_entry_sec = float(row['slp_T_Entry_time'])
                choice_sec = float(row['slp_choice_time'])

                if pd.isna(t_entry_sec) or pd.isna(tone_onset_sec) or pd.isna(choice_sec):
                    logger.warning('Warning: Skipping trial %s due to NaN in critical timing column(s).', trial_key)
                    drop_idxs.append(index)
                    continue

                # Calculate std_dev
                std_dev_sec = (choice_sec - t_entry_sec) / 3

                if std_dev_sec <= 1e-6:
                    std_dev_sec = self.params['min_experimental_std_dev']
                    logger.debug('Trial %s: choice interval <= 0, using min_experimental_std', trial_key)

                # Calculate bump_mean
                bump_peak_time_relative_to_onset_set = choice_sec - tone_onset_sec

                bump_peak_time_idx = np.argmin(np.abs(self.time - bump_peak_time_relative_to_onset_set))
                bump_peak_time_idx = np.clip(bump_peak_time_idx, 0, self.params['seq_len'] - 1)

                if bump_peak_time_idx >= DEFAULT_PARAMS['seq_len']:
                    drop_idxs.append(index)
                    continue

                peak_times.append(bump_peak_time_idx)

                std_dev_samples = int(round(std_dev_sec * self.params['sample_rate']))
                std_devs.append(std_dev_samples)

                # Determine zero_until time
                zero_until_time_relative_to_onset_sec = t_entry_sec - tone_onset_sec

                if zero_until_time_relative_to_onset_sec < 0:
                    zero_until_idx = 0
                else:
                    zero_until_idx = np.argmin(np.abs(self.time - zero_until_time_relative_to_onset_sec))
                    zero_until_idx = np.clip(zero_until_idx, 0, self.params['seq_len'] - 1)

                zero_until.append(zero_until_idx)

        self.df = self.df.drop(drop_idxs) # drop quitting trials
        self.df = self.df.assign(bump_peak_time_idx=peak_times,
                                 std_dev_samples=std_devs,
                                 zero_until_idx=zero_until)

    def __getitem__(self,
                    index,
                    p=None):
        '''
        Returns a single sample (input, target) pair

        Parameters
        ----------
            index : int, optional (ignored if self.generate)
            p : float
                Used if generate=True (primarily for plotting/testing)
        '''
        if self.generate:
            # --- Generation Mode ---
            if p is None:
                p = np.random.choice([0.0, 0.2, 0.8, 1.0])
            elif not 0.0 <= p <= 1.0:
                logger.warning('Provided p=%d is outside of [0, 1]', p)

            decision = 1 if np.random.rand() <= p else 0

            decision_time_idx = int(round(np.random.exponential(self.params['mean_duration'] * self.params['sample_rate'])))
            peak_time_idx = int(round(decision_time_idx + self.params['std_dev'] * self.params['sample_rate'] * 3))
            std_dev_samples = int(round(self.params['std_dev'] * self.params['sample_rate']))

            zero_until_idx = np.clip(decision_time_idx, 0, self.params['seq_len'] - 1)
            bump_peak_time_idx = np.clip(peak_time_idx, 0, self.params['seq_len'] - 1)

            return self._build_sample_tensors(p, decision, bump_peak_time_idx,
                                              std_dev_samples, zero_until_idx)
        else:
            # --- Experimental Data Loading Mode ---
            if not 0 <= index < self.num_samples:
                raise IndexError(f'Index {index} out of bounds for data of size {self.num_samples}.')

            try:
                trial_params = self.df.iloc[index]
                p_trial = trial_params['offer_prob'] / 100
                decision_trial = trial_params['slp_accept']
                bump_peak_time_idx_trial = trial_params['bump_peak_time_idx']
                std_dev_samples_trial = trial_params['std_dev_samples']
                zero_until_idx_trial = trial_params['zero_until_idx']
            except (KeyError, IndexError) as e:
                logger.error('Failed to retrieve parameters for index %d: %s', index, e)
                raise RuntimeError(f'Internal error retrieving trial parameters for index {index}') from e

            inputs_tensor, target_tensor = self._build_sample_tensors(p_trial, decision_trial, bump_peak_time_idx_trial,
                                              std_dev_samples_trial, zero_until_idx_trial)
            return inputs_tensor, target_tensor, p_trial

    def plot_trial(self,
                   index=None,
                   p=None):
        """
        Plot the input signal and target outputs for a single trial.

        Parameters
        ----------
            index : int, optional
                Used if generate=False
            p : int, optional
                Probability of reward for the sample trial
        """

        if self.generate:
            if index is not None:
                logger.warning("'index' argument ignored when generate=True")
            inputs, targets = self.__getitem__(index=0, p=p)
            title_suffix = f"(Generated, p={p if p is not None else 'random'})"
        else:
            if p is not None:
                logger.warning("'p' argument ignored when generate=False")
            plot_index = index if index is not None else 0
            if not 0 <= plot_index < self.num_samples:
                logger.error('Cannot plot trial: index %d out of bounds (%d trials loaded)', plot_index, self.num_samples)
                return

            try:
                inputs, targets, p_trial = self.__getitem__(index=plot_index)
                trial_info = self.df.iloc[plot_index]
                title_suffix = (f"(Experimental, Index: {plot_index}, Animal/Session: {trial_info['animal']}/{trial_info['session']})")
            except (IndexError, RuntimeError) as e:
                logger.error('Failed to get item for plotting at index %d: %s', plot_index, e)
                return

        inputs_np, targets_np = inputs.cpu().numpy(), targets.cpu().numpy()

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot input
        ax1.plot(self.time, inputs_np[:, 0], label='Input: Probability', color='blue')
        # ax1.plot(self.time, inputs_np[:, 1], label='Input: Positional Enc.', color='green', linestyle=':')
        ax1.plot(self.time, inputs_np[:, 1], label='Input: Ramp', color='red', linestyle='--') # change from index 2
        ax1.set_title(f'Input Signals {title_suffix}')
        ax1.set_ylabel("Value")
        ax1.set_ylim(-1.2, 1.2)
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # Plot target output
        ax2.plot(self.time, targets_np, color='orange', label='Target Output')
        ax2.set_title(f'Target Output {title_suffix}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Magnitude')
        ax2.set_ylim(-self.params['magnitude'] * 1.2, self.params['magnitude'] * 1.2)
        ax2.grid(True)
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
