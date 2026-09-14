# Synthetic line dataset and deterministic data-loader helpers.

import torch
import torch.utils.data as data


class LineDataset(data.Dataset):
    def __init__(self, num_samples=2000, img_size=3):
        self.data = []
        self.labels = []
        center = img_size // 2

        for idx in range(num_samples):
            img = torch.zeros((1, img_size, img_size))
            if idx % 2 == 0:
                # Class 0: vertical line in the middle column.
                img[0, :, center] = 1.0
                label = 0
            else:
                # Class 1: horizontal line in the middle row.
                img[0, center, :] = 1.0
                label = 1
            self.data.append(img)
            self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]