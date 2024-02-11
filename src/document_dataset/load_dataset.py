import tqdm
import json
import os
import PIL.Image
import copy

from .dataset import DocumentSample, DocumentDataset, DocumentSamplesList
from .encode_decode import normalize_boxes, resize_image

IOB2_TAG_FORMAT = "IOB2"
IOBES_TAG_FORMAT = "IOBES"

IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]
DATA_FORMAT = "json"

INFO_NAME = "name"
INFO_SPLITS = "splits"


def check_data_directory(data_directory):
    return os.path.isfile(data_directory)


def check_image_directory(image_directory):
    return os.path.isfile(image_directory)


def extract_words_boxes_labels_entities(data, tag_format):
    words, boxes, labels = [], [], []
    entities = {"start": [], "end": [], "label": []}
    entities_map = {}
    for i in range(len(data)):
        entities_map[data[i]["id"]] = len(entities["start"])
        start = len(words)
        words += data[i]["words"]
        end = len(words) - 1
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
        if label != "O":
            entities["start"].append(start)
            entities["end"].append(end)
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
    dataset_info_file = f"{dataset_directory}/dataset_info.json"
    with open(dataset_info_file, "r") as fp:
        dataset_info = json.load(fp)
    return dataset_info


def load_splits(samples, dataset_info):
    splits_info = copy.copy(dataset_info["splits"])
    dataset_splits = copy.copy(splits_info)

    def replace_id_by_sample(item):
        if isinstance(item, dict):
            for key in item.keys():
                item[key] = replace_id_by_sample(item[key])
        elif isinstance(item, list):
            for i in range(len(item)):
                item[i] = samples[item[i]]
            return DocumentSamplesList(item)
        return item

    replace_id_by_sample(dataset_splits)

    return dataset_splits


def load_dataset(dataset_directory, tag_format="IOB2", resize_images=True):
    document_samples = DocumentSamplesList()
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
                if check_data_directory(
                        data_directory) and check_image_directory(
                            image_directory):
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
            document_samples.append(sample)
            pbar.update()
    labels = document_samples.extract_samples_labels()
    dataset_info = load_dataset_info(dataset_directory)
    dataset_splits = load_splits(document_samples, dataset_info)
    dataset = DocumentDataset(name=dataset_info[INFO_NAME],
                              samples=document_samples,
                              splits=dataset_splits,
                              tag_format=tag_format,
                              labels=labels)
    return dataset
