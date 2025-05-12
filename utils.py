'''
This module provides utility functions that support the training and evaluation
of recurrent mixture density networks (R-MDNs), including data loading/generation
and visualization.

Methods
------
    DecisionDataset : class

    visualize_performance
        Parameters
        ----------
            model: torch.nn.Module
            dataset: DecisionDataset
            p_list : list of float
            n_samples : int

    visualize_mixture_components
        Parameters
        ----------
            model : torch.nn.Module
            dataset : DecisionDataset
            p_list : list of float
            n_components : int  

    visualize_manifold
        Parameters
        ----------
            model: torch.nn.Module
            dataset : DecisionDataset
            method : str ('pca' or 'tsne')
            p : float, optional

'''

# Imports

import glob
import logging
import os

import torch
import sklearn
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
    'animal', 'session', 'trial', 'tone_onset', 'tone_prob', 'T_Entry', 'choice', 'accept',
    # 'outcome', 'reward', 'quit', 'trial_end', 'exit' # Other potentially useful columns
]

# The directory of the experimental training data
DEFAULT_DATA_DIR = '../data'

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
        __init__(num_samples, params, generate, add_noise, 
                 animal, session, experimental_data_glob_pattern, data_dir)
            Usage
            -----
                * In order to load a single session: generate=False, 
                animal, session, experimental_data_glob_pattern, and data_dir must be provided
                * In order to load all sessions: generate=False
                experimental_data_glob_pattern, and data_dir must be provided
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
                 *, # Requires that additional arguments be passed by keyword only
                 num_samples=None,
                 params=None,
                 add_noise=False,
                 animal=None,
                 session=None,
                 experimental_data_glob_pattern='*/*/*RR*.csv',
                 data_dir=DEFAULT_DATA_DIR):
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
            animal : str, optional
                Used if generate=False and session is provided
            session : str, optional
                Used if generate=False and animal is provided
            experimental_data_glob_pattern : str
                Glob pattern relative to data_dir to find experimental data files
            data_dir : str, optional (required if generate=False)
        '''

        super().__init__()

        self.generate = generate
        self.data_dir = data_dir
        self.add_noise = add_noise
        self.experimental_trial_params = []

        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        for key in ['seq_len', 'sample_rate', 'noise_std', 'magnitude']:
            if key not in self.params:
                raise ValueError(f"Missing essential parameter '{key}' in parameter dictionary")

        self.time = np.arange(self.params['seq_len']) / self.params['sample_rate']

        if self.generate:
            # --- Generation Mode ---
            self.num_samples = num_samples if num_samples is not None else 10000
            if 'mean_duration' not in self.params or 'std_dev' not in self.params:
                raise ValueError("Missing 'mean_duration" or 'std_dev in params for generation mode.')

            logger.info("Initialized Dataset in GENERATE mode with %s samples.", self.num_samples)
        else:
            # --- Experimental Data Loading Mode ---
            logger.info('Initialized Dataset in LOAD mode.')
            self.num_samples = 0

            if animal is not None and session is not None:
                # Load a specific session
                target_glob_pattern = os.path.join(animal, session, '*RR*.csv')
                logger.info('Attempting to load specific session: %s/%s', animal, session)
            else:
                target_glob_pattern = experimental_data_glob_pattern
                logger.info("Attempting to load using glob pattern: '%s'", target_glob_pattern)

            logger.info('Searching in data directory: %s...', data_dir)

            try:
                all_trials_df = self._load_matching_files(
                    self.data_dir,
                    target_glob_pattern,
                    columns=REQUIRED_COLUMNS
                )

                if all_trials_df is None or all_trials_df.empty:
                    raise FileNotFoundError('No experimental data loaded or dataframe is empty.')
                
                self._preprocess_experimental_data(all_trials_df, columns=REQUIRED_COLUMNS)

                self.num_samples = len(self.experimental_trial_params)
                
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
            This logic is reused for both generated and experimental data (using their resp. parameters)

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
        positional_encoding = 0.25 * np.sin(2 * np.pi * self.time)

        # Feature 3
        ramp = self.time / self.time[-1] if self.params['seq_len'] > 1 and self.time[-1] > 0 else np.zeros_like(self.time)

        # Stack features
        inputs_np = np.stack((prob_input, positional_encoding, ramp), axis=1)
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
    
    def _load_matching_files(self, 
                             data_dir, 
                             glob_pattern, 
                             columns=None):
        '''
        Loads experimental data from all files matching the glob pattern within the data directory

        Parameters
        ----------
            data_dir : str
                Base directory
            glob_pattern : str
                Glob pattern relative to data_dir
            columns : list of str, optional
                Columns to load

        Returns
        -------
            combined_df : pandas.DataFrame
                Concatenated dataframe of all loaded data, or None if no files are found
        '''

        full_pattern = os.path.join(data_dir, glob_pattern)
        matching_files = sorted(glob.glob(full_pattern, recursive=True))

        if not matching_files:
            logger.warning('No files found matching pattern: %s', full_pattern)
            return None 
        
        logger.info('Found %d file(s) matching pattern.', len(matching_files))
        all_dfs = []

        for file_path in matching_files:
            logger.info('Loading file: %s', file_path)
            try:
                relative_path = os.path.relpath(file_path, data_dir)
                parts = relative_path.split(os.sep)

                if len(parts) >= 3:
                    animal_name = parts[-3]
                    session_name = parts[-2]
                else:
                    animal_name = 'unknown_animal'
                    session_name = 'unknown_session'
                    logger.warning('Warning: Could not reliably extract animal/session from path: %s', file_path)

                df = self._load_session_file(file_path, animal_name, session_name, columns=columns)

                if df is not None and not df.empty:
                    all_dfs.append(df)

            except (IOError, pd.errors.ParserError, ValueError, KeyError) as e:
                logger.error('Unexpected error processing file %s: %s', file_path, e)
        
        if not all_dfs:
            logger.error("No dataframes were successfully loaded from any matching file.")
            return None
        
        try:
            logger.info('Concatenating %d dataframes...', len(all_dfs))
            combined_df = pd.concat(all_dfs, ignore_index=True)
            return combined_df
        except ValueError as e:
            logger.error('Error concatenating dataframes: %s', e)
            return None
        
    def _load_session_file(self, 
                           filepath, 
                           animal_name, 
                           session_name, 
                           columns=None):
        '''
        Loads experimental trial data from a single CSV file
        Internal helper method

        Parameters
        ----------
            filepath : str
                Full path to CSV file
            animal_name : str
            session_name : str
            columns : list of str, optional
                Columns to load from CSV

        Returns
        -------
            df : pandas.DataFrame, or None if an error occurs during loading/validation
        '''

        if not os.path.exists(filepath):
            logger.error('Error: file not found at %s', filepath)
            return None
        
        try:
            df = pd.read_csv(filepath, usecols=columns if columns else None)

            if df.empty:
                logger.warning('File is empty: %s', filepath)

            if columns:
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    logger.error('Missing required columns in %s: %s', filepath, missing_cols)
                    return None

            df['animal'] = animal_name
            df['session'] = session_name
            return df
        
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            logger.error('An error occurred during the loading of session data: %s: %s', filepath, e)
            return None
        
    def _preprocess_experimental_data(self, 
                                      all_trials_df, 
                                      columns=None):
        '''
        Processes the combined dataframe into lists of parameters per trial
            Note: The tensor building happens on demand in __getitem__

        Parameters
        ----------
            all_trials_df : pandas.DataFrame
                Combined DataFrame of all loaded data
            columns : list of str, optional
                Columns to load from CSV
        '''

        required_grouping_cols = ['animal', 'session', 'trial']
        if not all(col in all_trials_df.columns for col in required_grouping_cols):
            missing = [col for col in required_grouping_cols if col not in all_trials_df.columns]
            raise ValueError('DataFrame missing required grouping columns for preprocessing: {missing}')

        if columns is None:
            columns = REQUIRED_COLUMNS

        try:        
            trial_groups = all_trials_df.groupby(required_grouping_cols, sort=False)
        except KeyError as e:
            logger.error("Grouping columns unexpectedly missing despite initial check.")
            raise ValueError("Failed to group DataFrame by 'animal', 'session', 'trial'") from e
        
        self.experimental_trial_params = []
        num_groups = len(trial_groups)
        logger.info('Processing %d unique trial group(s) from experimental data...', num_groups)
        processed_count, skipped_count = 0, 0

        for (animal_name, session_name, trial_id), trial_df in trial_groups:
            trial_key = f'({animal_name}, {session_name}, {trial_id})'

            if trial_df.empty:
                logger.warning('Warning: skipping empty group for trial %s.', trial_key)
                skipped_count += 1
                continue

            if not all(col in trial_df.columns for col in columns):
                missing = [col for col in columns if col not in trial_df.columns]
                logger.warning("Warning: Skipping trial %s due to missing columns: %s.", trial_key, missing)
                skipped_count += 1
                continue

            try:
                trial_data = trial_df.copy()

                p = float(trial_data['tone_prob'].iloc[0])
                decision = int(trial_df['accept'].iloc[0])
                tone_onset_sec = float(trial_df['tone_onset'].iloc[0])
                t_entry_sec = float(trial_df['T_Entry'].iloc[0])
                choice_sec = float(trial_df['choice'].iloc[0])

                if pd.isna(t_entry_sec) or pd.isna(tone_onset_sec) or pd.isna(choice_sec):
                    logger.warning('Warning: Skipping trial %s due to NaN in critical timing column(s).', trial_key)
                    skipped_count += 1
                    continue

                # Calculate std_dev 
                std_dev_sec = (choice_sec - t_entry_sec) / 3

                if std_dev_sec <= 1e-6:
                    std_dev_sec = self.params['min_experimental_std_dev']
                    logger.debug('Trial %s: choice interval <= 0, using min_experimental_std', trial_key)
                std_dev_samples = int(round(std_dev_sec * self.params['sample_rate']))
                
                # Calculate bump_mean
                bump_peak_time_relative_to_onset_set = choice_sec - tone_onset_sec
                
                if bump_peak_time_relative_to_onset_set < 0:
                    logger.warning('Calculated bump peak time is negative for trial %s.', trial_id)

                bump_peak_time_idx = np.argmin(np.abs(self.time - bump_peak_time_relative_to_onset_set))
                bump_peak_time_idx = np.clip(bump_peak_time_idx, 0, self.params['seq_len'] - 1)

                # Determine zero_until time
                zero_until_time_relative_to_onset_sec = t_entry_sec - tone_onset_sec

                if zero_until_time_relative_to_onset_sec < 0:
                    zero_until_idx = 0
                else:
                    zero_until_idx = np.argmin(np.abs(self.time - zero_until_time_relative_to_onset_sec))
                    zero_until_idx = np.clip(zero_until_idx, 0, self.params['seq_len'] - 1)

                self.experimental_trial_params.append({
                    'animal' : animal_name,
                    'session' : session_name,
                    'trial_id' : trial_id,
                    'p' : p,
                    'decision' : decision,
                    'bump_peak_time_idx' : int(bump_peak_time_idx),
                    'std_dev_samples' : float(std_dev_samples),
                    'zero_until_idx' : int(zero_until_idx)
                })
                processed_count += 1
            
            except (KeyError, ValueError, TypeError) as e:
                logger.error('Unexpected error occurred when processing trial %s: %s', trial_id, e)
                skipped_count += 1

        logger.info('Finished preprocessing. Processed: %d, Skipped: %d', processed_count, skipped_count)

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
            ## Generate Mode
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
            ## Load Mode
            if not 0 <= index < self.num_samples:
                raise IndexError(f'Index {index} out of bounds for data of size {self.num_samples}.')
            if not self.experimental_trial_params:
                raise RuntimeError('Dataset is in load mode but no experimental trial parameters were loaded.')
            
            try:
                trial_params = self.experimental_trial_params[index]
                p_trial = trial_params['p']
                decision_trial = trial_params['decision']
                bump_peak_time_idx_trial = trial_params['bump_peak_time_idx']
                std_dev_samples_trial = trial_params['std_dev_samples']
                zero_until_idx_trial = trial_params['zero_until_idx']
            except (KeyError, IndexError) as e:
                logger.error('Failed to retrieve parameters for index %d: %s', index, e)
                raise RuntimeError(f'Internal error retrieving trial parameters for index {index}') from e

            return self._build_sample_tensors(p_trial, decision_trial, bump_peak_time_idx_trial,
                                              std_dev_samples_trial, zero_until_idx_trial)
        
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
                inputs, targets = self.__getitem__(index=plot_index)
                trial_info = self.experimental_trial_params[plot_index]
                title_suffix = (f"(Experimental, Index: {plot_index}, Animal/Session: {trial_info['animal']}/{trial_info['session']})")
            except (IndexError, RuntimeError) as e:
                logger.error('Failed to get item for plotting at index %d: %s', plot_index, e)
                return

        inputs_np, targets_np = inputs.cpu().numpy(), targets.cpu().numpy()

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot input
        ax1.plot(self.time, inputs_np[:, 0], label='Input: Probability', color='blue')
        ax1.plot(self.time, inputs_np[:, 1], label='Input: Positional Enc.', color='green', linestyle=':')
        ax1.plot(self.time, inputs_np[:, 2], label='Input: Ramp', color='red', linestyle='--')
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


#######################################################
## Timeseries Visualization Functions                ##
#######################################################

def visualize_performance(model: torch.nn.Module,
                          dataset: DecisionDataset,
                          p_list=None,
                          n_samples=5):
    """
    Visualizes multiple model responses for different probability values.

    Parameters
    ----------
        model : PyTorch nn.Module
            Assumed to return (pi, mu, sigma, output), where output is the relevant tensor
        dataset : PyTorch Dataset (instance of DecisionDataset)
        p_list : float or list of float, optional
            List of probability values, or a single float.
            If None, probabilities are randomly selected.
        n : int, optional
            Number of samples per probability value.
    """

    model.eval()

    if not dataset.generate and p_list is not None:
        logger.warning("'p_list' is provided but dataset is in load mode")
        p_list = [None] * 1

    # Convert single float `p` to a list
    if p_list is None:
        p_values_to_plot = [None]  # Defaults to random probability selection
    elif isinstance(p_list, (float, int)):  # Single `p`
        p_values_to_plot = [float(p_list)]
    else:
        p_values_to_plot = [float(p) if p is not None else None for p in p_list]

    num_p_groups = len(p_values_to_plot)  # Number of probability values

    # Create a colormap for `n` different colors per probability value
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples)) # pylint: disable=no-member

    # Initialize figure with appropriate grid layout (rows = #p values, cols = 2: target + output)
    _, axes = plt.subplots(num_p_groups, 2, figsize=(12, 3 * num_p_groups), 
                           sharex=True, sharey=True, squeeze=False)

    time = dataset.time

    for p_idx, p_val in enumerate(p_values_to_plot):
        ax_target, ax_output = axes[p_idx, 0], axes[p_idx, 1]

        p_label = f'p = {p_val}' if p_val is not None else f"p={'random' if dataset.generate else 'loaded'}"

        # Label rows with corresponding `p` value
        ax_target.set_ylabel(p_label, fontsize=12, rotation=90, labelpad=15)

        # Set subplot titles (only for the first row)
        if p_idx == 0:
            ax_target.set_title("Target Outputs")
            ax_output.set_title("Model Outputs")

        # Keep y-axis limits consistent
        ax_target.set_ylim(-1.5, 1.5)
        ax_output.set_ylim(-1.5, 1.5)

        # Grid for clarity
        ax_target.grid(True)
        ax_output.grid(True)

        # Generate `n` trials for this probability `p`
        for i in range(n_samples):
            try:
                if dataset.generate:
                    inputs, targets = dataset.__getitem__(index=0, p=p_val)
                else:
                    random_index = np.random.randint(0, len(dataset))
                    inputs, targets = dataset.__getitem__(index=random_index)

                inputs = inputs.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(inputs)[-1]

                output_np = output.squeeze(0).cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Handle potential extra dimension in output
                if output_np.ndim == 2 and output_np.shape[-1] == 1:
                    output_np = output_np.squeeze(-1)

                # Handle potential mismatch in sequence length
                if output_np.shape[0] != dataset.params['seq_len']:
                    min_len = min(output_np.shape[0], dataset.params['seq_len'])
                    output_np = output_np[:min_len]
                    targets_np = targets_np[:min_len]
                    time_plot = time[:min_len]
                else:
                    time_plot = time

                label_suffix = f' Trial {i + 1}'
                ax_target.plot(time_plot, targets_np, color=colors[i], alpha=0.8, label='Target' + label_suffix)
                ax_output.plot(time_plot, output_np, color=colors[i], alpha=0.8, label='Output' + label_suffix)

            except IndexError as e:
                logger.error('IndexError during visualization sample %d: %s', i, e)
                break

            handles_target, labels_target = ax_target.get_legend_handles_labels()
            
            if handles_target:
                ax_target.legend([handles_target[0]], [labels_target[0].split(' Trial')[0]], loc='best')
            
            handles_output, labels_output = ax_output.get_legend_handles_labels()
            
            if handles_output:
                ax_output.legend([handles_output[0]], [labels_output[0].split(' Trial')[0]], loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
            
    plt.suptitle('Model Performance Comparison', fontsize=14)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def visualize_mixture_components(model : torch.nn.Module, 
                                 dataset : DecisionDataset, 
                                 p_list=None, 
                                 n_components=3):
    '''
    Visualizes the mixture density components output by the recurrent MDN

    Parameters
    ----------
        model : PyTorch nn.Module
        dataset : PyTorch Dataset (instance of DecisionDataset)
        p_list : float or list of float, optional
        n_components : int, optional
    '''

    model.eval()

    if not dataset.generate and p_list is not None:
        logger.warning("'p_list' is provided but dataset is in load mode.")
        p_list = [None]

    # Convert single float `p` to a list
    if p_list is None:
        p_values_to_plot = [None]  # Defaults to random probability selection
    elif isinstance(p_list, (float, int)):  # Single `p`
        p_values_to_plot = [float(p_list)]
    else:
        p_values_to_plot = [float(p) if p is not None else None for p in p_list]

    num_p_groups = len(p_values_to_plot)
    time = dataset.time

    if not n_components or n_components <= 0:
        logger.error('Invalid number of components. Cannot visualize')

    _, axes = plt.subplots(num_p_groups, n_components,  figsize=(4 * n_components, 3 * num_p_groups), 
                           sharex=True, sharey=False, squeeze=False)

    for p_idx, p_val in enumerate(p_values_to_plot):
        try:
            if dataset.generate:
                inputs, _ = dataset.__getitem__(index=0, p=p_val) # index ignored
            else:
                random_index = np.random.randint(0, len(dataset))
                inputs, _ = dataset.__getitem__(index=random_index)

            inputs = inputs.unsqueeze(0).to(device) # Add batch dim

            # Inference
            with torch.no_grad():
                # Assumes model returns (pi, mu, sigma, ...)
                output_tuple = model(inputs)

            if not (isinstance(output_tuple, (list, tuple)) and len(output_tuple) >= 3):
                logger.error("Model output format error for p=%d. Expected tuple/list length >= 3.", p_val)
                continue

            # Extract pi, mu, sigma (assuming order)
            pi = output_tuple[0]
            mu = output_tuple[1]
            sigma = output_tuple[2]

            # Validate shapes (expecting Batch, SeqLen, NComponents)
            expected_shape_part = (1, dataset.params['seq_len'], n_components)
            if pi.shape != expected_shape_part or mu.shape != expected_shape_part or sigma.shape != expected_shape_part:
                logger.warning("Unexpected component tensor shape for p=%d. ", p_val)
                continue

            # Convert tensors to NumPy arrays (remove batch dimension)
            pi_np = pi.squeeze(0).cpu().numpy()        # Shape: (SEQ_LEN, n_components)
            mu_np = mu.squeeze(0).cpu().numpy()        # Shape: (SEQ_LEN, n_components)
            sigma_np = sigma.squeeze(0).cpu().numpy()  # Shape: (SEQ_LEN, n_components)
            sigma_np = np.maximum(sigma_np, 1e-6)

            # Plot each component
            for k in range(n_components):
                ax = axes[p_idx, k]

                # Plot Mu and Sigma (mean +/- std dev)
                ax.plot(time, mu_np[:, k], label=rf'$\mu_{k+1}$', color='blue')
                ax.fill_between(time, mu_np[:, k] - sigma_np[:, k], mu_np[:, k] + sigma_np[:, k],
                                color='blue', alpha=0.3, label=rf'$\mu_{k+1} \pm \sigma_{k+1}$')

                # Plot Pi (mixture weight) on a secondary axis if scales differ significantly
                ax_pi = ax.twinx()
                ax_pi.plot(time, pi_np[:, k], label=rf'$\pi_{k+1}$', color='black', linestyle=':')
                ax_pi.set_ylabel(rf'$\pi_{k+1}$', color='black')
                ax_pi.set_ylim(0, 1.05) # Mixture weights should be in [0, 1]
                ax_pi.tick_params(axis='y', labelcolor='black')

                # --- Combine legends ---
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_pi.get_legend_handles_labels()
                ax_pi.legend(lines + lines2, labels + labels2, loc='upper right')

                # Labels and Grid
                ax.grid(True, axis='x') # Grid only on x-axis to reduce clutter
                ax.set_ylim(mu_np.min() - sigma_np.max(), mu_np.max() + sigma_np.max()) # Dynamic Y limits for Mu

                # Set Y label for Mu only on the first column
                if k == 0:
                    ax.set_ylabel(r"Value ($\mu \pm \sigma$)")

                # Set row label (p value) only on the first column's Y-axis label
                p_label = f"p = {p_val:.2f}" if p_val is not None else f"p = {'random' if dataset.generate else 'loaded'}"
                if k == 0:
                    ax.set_ylabel(rf"{p_label} Value ($\mu \pm \sigma$)", fontsize=10)

                # Set titles only for the first row
                if p_idx == 0:
                    ax.set_title(f"Component {k + 1}")

        except IndexError as e:
            logger.error("IndexError during mixture visualization for p=%d: %s. Dataset might be empty.", p_val, e)
            break

    # Set common X label only on the bottom axes
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.suptitle(r'Mixture Density Network Components (Top: $\pi$, Bottom: $\mu \pm \sigma$)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

#######################################################
## Manifold Visualization Functions                  ##
#######################################################

def visualize_manifold(model: torch.nn.Module,
                       dataset : DecisionDataset,
                       method='pca',
                       p=None):
    '''
    Visualize the hidden layer activations on a low dimensional space (e.g. manifold)

    Parameters
    ----------
        model : PyTorch nn.Module
        dataset : PyTorch Dataset
        method : str
            Accepts 'pca' and 'tsne'
        p : float
    '''

    model.eval()

    try:
        if dataset.generate:
            inputs, targets = dataset.__getitem__(index=0, p=p) # index ignored
            p_val_used = p if p is not None else "random"
        else:
            idx = np.random.randint(len(dataset))
            inputs, targets = dataset.__getitem__(index=idx)
            p_val_used = dataset.experimental_trial_params[idx]['p']

        inputs = inputs.unsqueeze(0).to(device) # Add batch dim

        with torch.no_grad():
            # Placeholder - replace with your actual model call:
            _, _, _, output_tensor, h_states_tensor = model(inputs)

            # --- Process Tensors ---
            # Expected h_states shape: (batch, seq_len, hidden_dim) -> needs (seq_len, hidden_dim)
            if h_states_tensor.ndim == 3 and h_states_tensor.shape[0] == 1:
                h_states_np = h_states_tensor.squeeze(0).cpu().numpy()
            elif h_states_tensor.ndim == 2: # If already (seq_len, hidden_dim)
                h_states_np = h_states_tensor.cpu().numpy()
            else:
                raise ValueError(f"Hidden states tensor has unexpected shape: {h_states_tensor.shape}. Expected (1, seq_len, hidden_dim) or (seq_len, hidden_dim).")

            # Expected output shape: (batch, seq_len, feature_dim) -> needs (seq_len,)
            if output_tensor.ndim == 3 and output_tensor.shape[0] == 1:
                output_np = output_tensor.squeeze(0).cpu().numpy()
            elif output_tensor.ndim == 2: # Assume (seq_len, features)
                output_np = output_tensor.cpu().numpy()
            else:
                raise ValueError(f"Output tensor has unexpected shape: {output_tensor.shape}. Expected (1, seq_len, features) or (seq_len, features).")

            # If output has a feature dim of 1, squeeze it
            if output_np.ndim == 2 and output_np.shape[-1] == 1:
                output_np = output_np.squeeze(-1)
            elif output_np.ndim > 1:
                logger.warning("Model output has multiple features (%s). Plotting only the first feature.", output_np.shape[-1])
                output_np = output_np[:, 0]

    except (IndexError, RuntimeError, ValueError) as e:
        logger.error("Failed during data loading or model inference for manifold visualization: %s", e)
        raise RuntimeError("Could not get hidden states from model.") from e
    except AttributeError as e:
        logger.error("AttributeError during model call - does your model support returning hidden states correctly? Error: %s", e)
        raise RuntimeError("Model does not seem to support returning hidden states as expected.") from e

    # --- Perform Dimensionality Reduction ---
    logger.info("Performing %s dimensionality reduction on hidden states", method.upper(), )
    try:
        if method.lower() == 'pca':
            reducer = sklearn.decomposition.PCA(n_components=3)
            hidden_state_dynamics = reducer.fit_transform(h_states_np)
        elif method.lower() == 'tsne':
            reducer = sklearn.manifold.TSNE(n_components=3, learning_rate='auto', init='random', perplexity=max(5,min(30, h_states_np.shape[0]-1)) ) # Adjust perplexity
            hidden_state_dynamics = reducer.fit_transform(h_states_np)
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'pca' or 'tsne'.")
    except Exception as e:
        logger.exception("Error during {method.upper()} fitting/transforming: %s", e)
        raise RuntimeError(f"{method.upper()} failed.") from e

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 7))
    time_vector = dataset.time

    # Ensure output length matches time vector length (handle potential model output differences)
    output_len = len(output_np)
    if output_len != len(time_vector):
        logger.warning("Manifold viz: Output length (%d) differs from time vector length (%d).", output_len, len(time_vector))
        min_len = min(output_len, len(time_vector))
        output_np = output_np[:min_len]
        time_vector_plot = time_vector[:min_len]
        hidden_state_dynamics = hidden_state_dynamics[:min_len] # Also truncate manifold
    else:
        time_vector_plot = time_vector

    # Subplot 1: Model Output vs Time
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(time_vector_plot, output_np, label='Model Output', color='orange')
    targets_np = targets.cpu().numpy()
    if len(targets_np) == len(time_vector_plot): # Check length match
        ax1.plot(time_vector_plot, targets_np, label='Target', color='grey', linestyle=':')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'Model Output (p={p_val_used:.2f})')
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: 3D Manifold Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scatter = ax2.scatter(hidden_state_dynamics[:, 0], hidden_state_dynamics[:, 1], 
                          hidden_state_dynamics[:, 2], c=time_vector_plot, cmap='viridis', s=10)
    
    # Add start and end markers
    ax2.scatter(hidden_state_dynamics[0, 0], hidden_state_dynamics[0, 1], hidden_state_dynamics[0, 2],
                 c='red', s=100, marker='o', label='Start')
    ax2.scatter(hidden_state_dynamics[-1, 0], hidden_state_dynamics[-1, 1], hidden_state_dynamics[-1, 2],
                 c='black', s=100, marker='x', label='End')

    ax2.set_xlabel(f'{method.upper()} Comp 1')
    ax2.set_ylabel(f'{method.upper()} Comp 2')
    ax2.set_zlabel(f'{method.upper()} Comp 3')
    ax2.set_title(f'Hidden State Dynamics ({method.upper()})')
    ax2.legend()

    # Add a colorbar to show time mapping
    fig.colorbar(scatter, ax=ax2, shrink=0.6, aspect=20, label='Time (s)')
    plt.suptitle('Model Output and Hidden State Manifold Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()