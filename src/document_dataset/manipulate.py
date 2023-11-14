import random
import copy

from .dataset import DocumentDataset


def shuffle_dataset(dataset: DocumentDataset) -> DocumentDataset:
    samples = copy.copy(dataset.samples)
    random.shuffle(samples)
    shuffled_dataset = copy.copy(dataset)
    shuffled_dataset.samples = samples
    return shuffled_dataset


def k_fold_split(dataset: DocumentDataset, k: int) -> list[DocumentDataset]:
    n, m = divmod(len(dataset.samples), k)
    splited_samples = list(dataset.samples[i * n + min(i, m):(i + 1) * n +
                                           min(i + 1, m)] for i in range(k))
    partitions = list()
    for i in range(k):
        partition = copy.copy(dataset)
        partition.samples = splited_samples[i]
        partitions.append(partition)
    return partitions


def join_partitions(partitions: list[DocumentDataset],
                    partitions_index: list[int]) -> DocumentDataset:
    samples = list()
    for i in partitions_index:
        samples += partitions[i].samples
    dataset = copy.copy(partitions[0])
    dataset.samples = samples
    return dataset
