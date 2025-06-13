'''
This module provides utility functions that support the training and evaluation
of recurrent mixture density networks (R-MDNs), including time series and lower-
dimensional manifold visualization.

Methods
------
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
import logging

import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset from decision_dataset.py
import decision_dataset

# Basic logging configuration (logs INFO and above to console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device: %s.', device)

#######################################################
## Timeseries Visualization Functions                ##
#######################################################

def visualize_performance(model: torch.nn.Module,
                          dataset: decision_dataset.DecisionDataset,
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
    p_val_dict = {0.0 : 0, 0.2 : 1, 0.8 : 2, 1.0 : 3}

    if not dataset.generate and p_list is not None:
        logger.warning("'p_list' is provided but dataset is in load mode")
        p_list = None

    if p_list is None:
        p_values_to_plot = [None]  # Defaults to random probability selection
    elif isinstance(p_list, (float, int)):  # Single `p`
        p_values_to_plot = [float(p_list)]
    else:
        p_values_to_plot = [float(p) if p is not None else None for p in p_list]

    num_p_groups = len(p_values_to_plot)  # Number of probability values

    # Create a colormap for `n` different colors per probability value
    if dataset.generate:
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples)) # pylint: disable=no-member
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, 4)) # pylint: disable=no-member | 4 colors -- 4 probability values

    # Initialize figure with appropriate grid layout (rows = #p values, cols = 2: target + output)
    _, axes = plt.subplots(num_p_groups, 2, figsize=(12, 3 * num_p_groups),
                           sharex=True, sharey=True, squeeze=False)

    time = dataset.time

    p_colors_seen = {}

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
                    inputs, targets, p_val = dataset.__getitem__(index=random_index)

                inputs = inputs.unsqueeze(0).to(device)

                with torch.no_grad():
                    _, _, _, output = model(inputs, train=False)

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

                if dataset.generate:
                    label_suffix = f' Trial {i + 1}'
                    ax_target.plot(time_plot, targets_np, color=colors[i], alpha=0.8, label='Target' + label_suffix)
                    ax_output.plot(time_plot, output_np, color=colors[i], alpha=0.8, label='Output' + label_suffix)
                else:
                    color_idx = p_val_dict[p_val]
                    color=colors[color_idx]

                    # Store this p_value and color for legend
                    if p_val not in p_colors_seen:
                        p_colors_seen[p_val] = color

                    # Only add label for the first occurrence of each p_value
                    existing_p_vals_target = [line.get_label().split('(p=')[1].split(')')[0] for line in ax_target.get_lines() if '(p=' in line.get_label()]
                    existing_p_vals_output = [line.get_label().split('(p=')[1].split(')')[0] for line in ax_output.get_lines() if '(p=' in line.get_label()]
                    
                    target_label = f'Target (p={p_val})' if str(p_val) not in existing_p_vals_target else None
                    output_label = f'Output (p={p_val})' if str(p_val) not in existing_p_vals_output else None
                    
                    ax_target.plot(time_plot, targets_np, color=color, alpha=0.8, 
                                 label=target_label if target_label else None)
                    ax_output.plot(time_plot, output_np, color=color, alpha=0.8, 
                                 label=output_label if output_label else None)

            except IndexError as e:
                logger.error('IndexError during visualization sample %d: %s', i, e)
                break

            # Handle legends
            if dataset.generate:
                handles_target, labels_target = ax_target.get_legend_handles_labels()
                if handles_target:
                    ax_target.legend([handles_target[0]], [labels_target[0].split(' Trial')[0]], loc='best')

                handles_output, labels_output = ax_output.get_legend_handles_labels()
                if handles_output:
                    ax_output.legend([handles_output[0]], [labels_output[0].split(' Trial')[0]], loc='best')
            else:
                # For non-generate case, show legend with probability values and colors
                handles_target, labels_target = ax_target.get_legend_handles_labels()
                if handles_target:
                    # Get unique labels to avoid duplicates
                    unique_labels = []
                    unique_handles = []
                    seen_p_vals = set()
                    for handle, label in zip(handles_target, labels_target):
                        if '(p=' in label:
                            p_val_str = label.split('(p=')[1].split(')')[0]
                            if p_val_str not in seen_p_vals:
                                unique_handles.append(handle)
                                unique_labels.append(label)
                                seen_p_vals.add(p_val_str)
                    ax_target.legend(unique_handles, unique_labels, loc='best')

                handles_output, labels_output = ax_output.get_legend_handles_labels()
                if handles_output:
                    # Get unique labels to avoid duplicates
                    unique_labels = []
                    unique_handles = []
                    seen_p_vals = set()
                    for handle, label in zip(handles_output, labels_output):
                        if '(p=' in label:
                            p_val_str = label.split('(p=')[1].split(')')[0]
                            if p_val_str not in seen_p_vals:
                                unique_handles.append(handle)
                                unique_labels.append(label)
                                seen_p_vals.add(p_val_str)
                    ax_output.legend(unique_handles, unique_labels, loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.suptitle('Model Performance Comparison', fontsize=14)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def visualize_mixture_components(model : torch.nn.Module,
                                 dataset : decision_dataset.DecisionDataset,
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

    _, axes = plt.subplots(num_p_groups, n_components,
                           figsize=(4 * n_components, 3 * num_p_groups),
                           sharex=True, sharey=False, squeeze=False)

    for p_idx, p_val in enumerate(p_values_to_plot):
        try:
            if dataset.generate:
                inputs, _ = dataset.__getitem__(index=0, p=p_val) # index ignored
            else:
                random_index = np.random.randint(0, len(dataset))
                inputs, _, p_val = dataset.__getitem__(index=random_index)

            inputs = inputs.unsqueeze(0).to(device) # Add batch dim

            # Inference
            with torch.no_grad():
                # Assumes model returns (pi, mu, sigma, ...)
                output_tuple = model(inputs, train=False)

            if not (isinstance(output_tuple, (list, tuple)) and len(output_tuple) >= 3):
                logger.error("Model output format error for p=%d. Expected tuple/list length >= 3.", p_val)
                continue

            # Extract pi, mu, sigma (assuming order)
            pi = output_tuple[0]
            mu = output_tuple[1]
            sigma = output_tuple[2]

            # Validate shapes (expecting Batch, SeqLen, NComponents)
            # expected_shape_part = (1, dataset.params['seq_len'] - 1, n_components)
            # if pi.shape != expected_shape_part or mu.shape != expected_shape_part or sigma.shape != expected_shape_part:
            #     logger.warning("Unexpected component tensor shape for p=%d. ", p_val)
            #     continue

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
                       dataset : decision_dataset.DecisionDataset,
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
            # TODO: update to include UMAP
        p : float
    '''

    model.eval()

    try:
        if dataset.generate:
            inputs, targets = dataset.__getitem__(index=0, p=p) # index ignored
            p_val_used = p if p is not None else "random"
        else:
            idx = np.random.randint(len(dataset))
            inputs, targets, p_val_used = dataset.__getitem__(index=idx)
            p_val_used = dataset.experimental_trial_params[idx]['p']

        inputs = inputs.unsqueeze(0).to(device) # Add batch dim

        with torch.no_grad():
            # Placeholder - replace with your actual model call:
            _, _, _, output_tensor, h_states_tensor = model(inputs, train=False, return_hidden=True)

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
    ax2.scatter(hidden_state_dynamics[0, 0], hidden_state_dynamics[0, 1], 
                hidden_state_dynamics[0, 2], c='red', s=100, marker='o', label='Start')
    ax2.scatter(hidden_state_dynamics[-1, 0], hidden_state_dynamics[-1, 1], 
                hidden_state_dynamics[-1, 2], c='black', s=100, marker='x', label='End')

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
    