import tqdm
import json
import os
import PIL.Image
import copy

from .dataset import DocumentSample, DocumentDataset
from .encode_decode import normalize_boxes, resize_image

IOB2_TAG_FORMAT = "IOB2"
IOBES_TAG_FORMAT = "IOBES"

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]
DATA_FORMAT = "json"

INFO_NAME = "name"
INFO_SPLITS = "splits"
INFO_CITATION = "citation"


def extract_labels(samples: dict[str, DocumentSample]):
    labels_tags = set()
    entities_labels_tags = set()
    prefixes = set()
    for sample in samples.values():
        for label in sample.entities["label"]:
            entities_labels_tags.add(label)
        for label in sample.labels:
            if label == "O":
                continue
            prefix, label = label.split("-")
            prefixes.add(prefix)
            labels_tags.add(label)
    prefixes = sorted(list(prefixes))
    labels_tags = list(labels_tags)
    re_labels = list(entities_labels_tags)
    tc_labels = ["O"]
    for label in labels_tags:
        for prefix in prefixes:
            tc_labels.append(prefix + "-" + label)
    return tc_labels, re_labels


def create_label2id(labels: list[str]) -> dict[str, int]:
    label2id: dict[str, int] = {}
    for i in range(len(labels)):
        label2id[labels[i]] = i
    return label2id


def create_id2label(labels: list[str]) -> dict[int, str]:
    id2label: dict[int, str] = {}
    for i in range(len(labels)):
        id2label[i] = labels[i]
    return id2label


def check_data(data_directory):
    return os.path.isfile(data_directory)


def check_image(image_directory):
    return os.path.isfile(image_directory)


def extract_words_boxes_labels_entities(data, tag_format):
    words, boxes, labels = [], [], []
    entities = {"start": [], "end": [], "label": []}
    entities_map = {}
    for i in range(len(data)):
        entities_map[data[i]["id"]] = len(entities["start"])
        entities["start"].append(len(words))
        words += data[i]["words"]
        entities["end"].append(len(words) - 1)
        boxes += data[i]["boxes"]
        for j in range(len(data[i]["words"])):
            if data[i]["label"] == "OTHER":
                labels.append("O")
            else:
                if tag_format == IOB2_TAG_FORMAT:
                    if j == 0:
                        labels.append("B-" + data[i]["label"])
                    else:
                        labels.append("I-" + data[i]["label"])
                elif tag_format == IOBES_TAG_FORMAT:
                    if len(data[i]["words"]) == 1:
                        labels.append("S-" + data[i]["label"])
                    elif j == 0:
                        labels.append("B-" + data[i]["label"])
                    elif j < len(data[i]["words"]) - 1:
                        labels.append("I-" + data[i]["label"])
                    elif j == len(data[i]["words"]) - 1:
                        labels.append("E-" + data[i]["label"])
        label = data[i]["label"]
        label = "O" if label == "OTHER" else label
        entities["label"].append(label)
    return words, boxes, labels, entities, entities_map


def extract_relations(data, entities, entities_map):
    relations = {"head": [], "tail": [], "start_index": [], "end_index": []}

    for i in range(len(data)):
        for link in data[i]["links"]:
            if not link:
                continue
            x, y = entities_map[link[0]], entities_map[link[1]]
            if x < len(entities["start"]) and y < len(entities["end"]):
                relations["head"].append(x)
                relations["tail"].append(y)
                start_index = min(entities["start"][x], entities["start"][y])
                relations["start_index"].append(start_index)
                end_index = max(entities["end"][x], entities["end"][y])
                relations["end_index"].append(end_index)
    return relations


def load_sample(data_directory,
                image_directory,
                tag_format=IOB2_TAG_FORMAT,
                id="",
                resize_images=True):
    with open(data_directory) as data_json:
        data = json.load(data_json)
    words, boxes, labels, entities, entities_map = extract_words_boxes_labels_entities(
        data, tag_format)
    relations = extract_relations(data, entities, entities_map)
    image = PIL.Image.open(image_directory).convert("RGB")
    if resize_images:
        boxes = normalize_boxes(boxes, image)
        image = resize_image(image)
    sample = DocumentSample(id=id,
                             words=words,
                             boxes=boxes,
                             labels=labels,
                             entities=entities,
                             relations=relations,
                             image=image)
    return sample


def load_dataset_info(dataset_directory) -> dict:
    normalized_path = os.path.normpath(dataset_directory)
    _, dataset_name = os.path.split(normalized_path)
    dataset_info_file = f"{dataset_directory}/{dataset_name}.json"
    with open(dataset_info_file, "r") as fp:
        dataset_info = json.load(fp)
    return dataset_info


def get_dataset_splits(samples: dict[str, DocumentSample], splits_info: dict):

    def substitute_id_with_sample_object(splits):
        if isinstance(splits, dict):
            for key, value in splits.items():
                if isinstance(value, (dict, list)):
                    splits[key] = substitute_id_with_sample_object(value)
                elif isinstance(value, str):
                    splits[key] = samples[value]
        elif isinstance(splits, list):
            split_samples = dict()
            for i, item in enumerate(splits):
                split_samples[item] = samples[item]
            splits = split_samples
        return splits

    splits = copy.copy(splits_info)
    splits = substitute_id_with_sample_object(splits)
    return splits


def load_dataset(dataset_directory, tag_format="IOB2", resize_images=True):
    samples = dict()
    datas_directory = f"{dataset_directory}/data/"
    images_directory = f"{dataset_directory}/image/"
    data_files = sorted(os.listdir(datas_directory))
    with tqdm.tqdm(desc="Loading dataset", total=len(data_files)) as pbar:
        for data_file in data_files:
            id, _ = data_file.split(".")
            image_directory = None
            data_directory = f"{datas_directory}/{id}.{DATA_FORMAT}"
            for image_extension in IMAGE_EXTENSIONS:
                image_directory = f"{images_directory}/{id}.{image_extension}"
                if check_data(data_directory) and check_image(image_directory):
                    break
            if image_directory is None:
                raise BaseException(
                    f"Image format not supported! Must be one of {IMAGE_EXTENSIONS}"
                )
            sample = load_sample(data_directory=data_directory,
                                 image_directory=image_directory,
                                 tag_format=tag_format,
                                 id=id,
                                 resize_images=resize_images)
            samples[sample.id] = sample
            pbar.update()
    tc_labels, re_labels = extract_labels(samples)
    dataset_info = load_dataset_info(dataset_directory)
    dataset_splits = get_dataset_splits(samples,
                                        splits_info=dataset_info["splits"])
    dataset = DocumentDataset(name=dataset_info[INFO_NAME],
                               splits=dataset_splits,
                               tag_format=tag_format,
                               tc_labels=tc_labels,
                               tc_label2id=create_label2id(tc_labels),
                               tc_id2label=create_id2label(tc_labels),
                               re_labels=re_labels,
                               re_label2id=create_label2id(re_labels),
                               re_id2label=create_id2label(re_labels),
                               citation=dataset_info[INFO_CITATION])
    return dataset
