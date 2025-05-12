import torch
from torch.utils.data import Dataset
import numpy as np

SAMPLE_RATE = 60
SEQ_LEN = 5 * SAMPLE_RATE # Hz
MEAN_DURATION = 0.25
STD_DEV = 0.25
NOISE_STD = 0.05
MAGNITUDE = 1.0

class TrialDataset(Dataset):
    def __init__(self, num_samples, seq_len=SEQ_LEN, sample_rate=SAMPLE_RATE, 
                 mean_duration=MEAN_DURATION, std_dev=STD_DEV, noise_std=NOISE_STD):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.mean_duration = mean_duration
        self.std_dev = std_dev
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Probability p
        p = np.random.choice([0.0, 0.2, 0.8, 1.0])
        decision = 1 if np.random.rand() <= p else 0

        # 2) Time axis
        time = np.arange(self.seq_len) / self.sample_rate

        # 3) Input features
        prob_input = np.full(self.seq_len, p - 0.5)
        positional_encoding = 0.25 * np.sin(2 * np.pi * time)
        ramp = time / time[-1]

        # Shape: (seq_len, 2)
        inputs_np = np.stack((prob_input, positional_encoding, ramp), axis=1)
        inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32)
        #  ^ NO unsqueeze(0)!

        # 4) Decision time
        decision_time = int(np.random.exponential(self.mean_duration) * self.sample_rate) + 20
        decision_time += np.random.randint(-5, 5)
        decision_time = np.clip(decision_time, 0, self.seq_len)

        # 5) Gaussian bump
        bump_center_sec = (decision_time / self.sample_rate) + 3 * self.std_dev
        bump_np = np.exp(-0.5 * ((time - bump_center_sec) / self.std_dev) ** 2)
        if bump_np.max() > 0:
            bump_np /= bump_np.max()

        target_np = bump_np * MAGNITUDE
        target_np[:decision_time] = 0.0
        if decision == 0:
            target_np *= -1.0

        # shape: (seq_len,) -> make it (seq_len, 1)
        target_tensor = torch.tensor(target_np, dtype=torch.float32).unsqueeze(-1)

        if self.noise_std > 0:
            inputs_noise = torch.randn_like(inputs_tensor) * self.noise_std / 2
            inputs_tensor += inputs_noise
            target_noise = torch.randn_like(target_tensor) * self.noise_std
            target_tensor += target_noise

        return inputs_tensor, target_tensor