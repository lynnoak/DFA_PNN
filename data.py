# Synthetic line dataset and deterministic data-loader helpers.

import torch
import torch.utils.data as data


class LineDataset(data.Dataset):
    def __init__(
        self,
        split="train",
        dataset_mode="clean",
        img_size=3,
        seed=0,
        train_corruptions_per_prototype=2,
        test_corruptions_per_class=3,
    ):
        if img_size != 3:
            raise ValueError("The line dataset requires img_size=3")
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if dataset_mode not in {"clean", "generalization"}:
            raise ValueError(f"Unsupported dataset mode: {dataset_mode}")

        prototypes = [(0, column) for column in range(img_size)]
        prototypes.extend((1, row) for row in range(img_size))
        if dataset_mode == "clean":
            patterns = self._clean_patterns(split, prototypes, img_size, seed)
        else:
            patterns = self._generalization_patterns(
                split,
                prototypes,
                img_size,
                seed,
                train_corruptions_per_prototype,
                test_corruptions_per_class,
            )

        self.data = torch.stack([image for image, _ in patterns])
        self.labels = torch.tensor([label for _, label in patterns], dtype=torch.long)

    @classmethod
    def _clean_patterns(cls, split, prototypes, img_size, seed):
        if split == "train":
            selected = prototypes
        else:
            generator = torch.Generator().manual_seed(seed)
            vertical_column = int(torch.randint(img_size, (1,), generator=generator).item())
            horizontal_row = int(torch.randint(img_size, (1,), generator=generator).item())
            selected = [(0, vertical_column), (1, horizontal_row)]
        return [(cls._make_pattern(label, position, img_size), label) for label, position in selected]

    @classmethod
    def _generalization_patterns(
        cls,
        split,
        prototypes,
        img_size,
        seed,
        train_corruptions_per_prototype,
        test_corruptions_per_class,
    ):
        generator = torch.Generator().manual_seed(seed)
        clean_patterns = [(cls._make_pattern(label, position, img_size), label) for label, position in prototypes]
        train_corruptions = []
        train_signatures = set()
        for label, position in prototypes:
            clean_image = cls._make_pattern(label, position, img_size)
            pixel_order = torch.randperm(img_size * img_size, generator=generator).tolist()
            for pixel_index in pixel_order[:train_corruptions_per_prototype]:
                corrupted = cls._flip_pixel(clean_image, pixel_index)
                train_corruptions.append((corrupted, label))
                train_signatures.add(cls._signature(corrupted))

        if split == "train":
            return clean_patterns + train_corruptions

        test_candidates = {0: [], 1: []}
        for label, position in prototypes:
            clean_image = cls._make_pattern(label, position, img_size)
            for pixel_index in range(img_size * img_size):
                corrupted = cls._flip_pixel(clean_image, pixel_index)
                if cls._signature(corrupted) not in train_signatures:
                    test_candidates[label].append(corrupted)

        selected = []
        for label in (0, 1):
            candidate_order = torch.randperm(len(test_candidates[label]), generator=generator).tolist()
            for candidate_index in candidate_order[:test_corruptions_per_class]:
                selected.append((test_candidates[label][candidate_index], label))
        return selected

    @staticmethod
    def _flip_pixel(image, pixel_index):
        corrupted = image.clone()
        row, column = divmod(pixel_index, image.shape[-1])
        corrupted[0, row, column] = 1.0 - corrupted[0, row, column]
        return corrupted

    @staticmethod
    def _signature(image):
        return tuple(int(value) for value in image.reshape(-1).tolist())

    @staticmethod
    def _make_pattern(label, position, img_size):
        image = torch.zeros((1, img_size, img_size))
        if label == 0:
            image[0, :, position] = 1.0
        else:
            image[0, position, :] = 1.0
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]