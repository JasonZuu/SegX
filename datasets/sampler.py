import torch
from torch.utils.data import Sampler
import numpy as np


class BalancedLabelSampler(Sampler):
    def __init__(self, labels: np.array, num_classes:int, batch_size:int, used_ratio=1.0):
        """
        This sampler ensures that each batch contains samples from all label classes.
        """
        assert batch_size > 1, "batch_size must be greater than 1"

        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.used_ratio = used_ratio

        self.samples_per_class = batch_size // num_classes
        self.extra_samples = batch_size % num_classes

        self.class_indices = [np.where(self.labels == i)[0] for i in range(num_classes)]

        for i in self.class_indices:
            assert len(i) >= self.samples_per_class, \
                f"Class has less than {self.samples_per_class} samples"

        self.used_label_indices_count = [0] * self.num_classes
        self.count = 0
        self.length = len(self.labels) // self.batch_size
        self.length = int(self.length * self.used_ratio)

    def __iter__(self):
        self.count = 0
        while self.count < self.length:
            indices = []
            extra_samples_allocated = 0

            for i in range(self.num_classes):
                n_samples = self.samples_per_class + (1 if extra_samples_allocated < self.extra_samples else 0)

                start_idx = self.used_label_indices_count[i]
                end_idx = start_idx + n_samples
                idx = self.class_indices[i][start_idx: end_idx]
                indices.extend(idx)
                self.used_label_indices_count[i] += n_samples

                if self.used_label_indices_count[i] + self.samples_per_class > len(self.class_indices[i]):
                    np.random.shuffle(self.class_indices[i])
                    self.used_label_indices_count[i] = 0

                if extra_samples_allocated < self.extra_samples:
                    extra_samples_allocated += 1

            yield indices
            self.count += 1

    def __len__(self):
        return self.length


class RandomLabelSampler(Sampler):
    def __init__(self, labels: np.array, num_classes: int, batch_size: int, used_ratio=1.0):
        """
        This sampler ensures that each batch contains samples from all label classes.
        :param labels: Array of labels.
        :param num_classes: Number of classes.
        :param batch_size: Size of each batch.
        """
        assert batch_size >= num_classes, "batch_size must be at least equal to num_classes"

        self.labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.used_ratio = used_ratio

        self.indices = np.arange(len(self.labels))
        np.random.shuffle(self.indices)
        self.class_indices = [np.where(self.labels == i)[0] for i in range(num_classes)]
        for i in self.class_indices:
            assert len(i) > 0, "Class must have at least one sample"

        self.length = len(self.labels) // self.batch_size - (self.num_classes - 1)
        self.length = int(self.length * self.used_ratio)

    def __iter__(self):
        self.count = 0
        while self.count < self.length:
            indices = []
            pomp_indices_i = []

            # Sequentially select batch_size - (num_classes - 1) samples
            seq_count = self.batch_size - (self.num_classes - 1)
            for i, index in enumerate(self.indices):
                indices.append(index)
                pomp_indices_i.append(i)
                seq_count -= 1
                if seq_count <= 0:
                    break

            # delete pomp_indices_i from self.indices
            self.indices = np.delete(self.indices, pomp_indices_i)

            # Check for missing classes
            existing_classes = set(self.labels[indices])
            missing_classes = set(range(self.num_classes)) - existing_classes

            # Fill the batch with missing classes or randomly if all classes are present
            for missing_class in missing_classes:
                idx = np.random.choice(self.class_indices[missing_class], 1)
                indices.extend(idx)

            if len(indices) < self.batch_size:
                additional_indices = np.random.choice(np.concatenate(self.class_indices), 
                                                      self.batch_size - len(indices), replace=False)
                indices.extend(additional_indices)

            yield indices
            self.count += 1

    def __len__(self):
        return self.length

