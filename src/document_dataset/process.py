import typing

import tqdm
import transformers

from .dataset import DocumentSamplesList, DocumentDataset, DocumentSample, EncodedDocumentSample
from .encode_decode import normalize_boxes, labels_to_ids, encode_image

MAX_LENGHT = 512
PADDING = "max_length"
PAD_TO_MULTIPLE_OF = 8
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
TRUNCATION = True
CLS_LABEL = 0
PAD_LABEL = 0
SEP_LABEL = 0
CLS_TOKEN_BOX = [0, 0, 0, 0]
SEP_TOKEN_BOX = [1000, 1000, 1000, 1000]
PAD_TOKEN_BOX = [1000, 1000, 1000, 1000]


def process_words_boxes_labels(sample,
                               tokenizer,
                               label2id,
                               max_lenght=MAX_LENGHT,
                               lowercase_all_words=False):
    input_ids = []
    bbox = []
    processed_labels = []
    attention_mask = []
    words2input_ids = {}

    words = sample.words
    boxes = sample.boxes
    labels = sample.labels
    image = sample.image
    entities = sample.entities

    if lowercase_all_words:
        words = [word.lower() for word in words]

    max_lenght_without_special = max_lenght - 2

    for start, end in zip(entities["start"], entities["end"]):
        for i in range(start, end + 1):
            word = words[i]
            label = labels[i]
            box = boxes[i]
            tokens = tokenizer.tokenize(word)
            if len(input_ids) + len(tokens) <= max_lenght_without_special:
                input_ids += tokenizer.convert_tokens_to_ids(tokens)
                bbox += normalize_boxes([box] * len(tokens), image)
                processed_labels += [label2id[label]] * len(tokens)
                attention_mask += [1] * len(tokens)
                words2input_ids[i] = (len(input_ids) - len(tokens),
                                      len(input_ids) - 1)

    # postprocessing
    while (len(input_ids) != max_lenght_without_special):
        input_ids.append(tokenizer.pad_token_id)
        bbox.append(PAD_TOKEN_BOX)
        processed_labels.append(PAD_LABEL)
        attention_mask.append(0)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    bbox = [CLS_TOKEN_BOX] + bbox + [SEP_TOKEN_BOX]
    processed_labels = [CLS_LABEL] + processed_labels + [SEP_LABEL]
    attention_mask = [0] + attention_mask + [0]

    return input_ids, bbox, processed_labels, attention_mask, words2input_ids


def process_entities_relations(sample, words2input_ids, label2id,
                               labels_to_exclude):

    if labels_to_exclude is None:
        labels_to_exclude = set()

    processed_entities = {"start": [], "end": [], "label": []}
    processed_relations = {
        "head": [],
        "tail": [],
        "start_index": [],
        "end_index": []
    }

    relations_pairs = set()

    entitie2processed_entities = {}

    entities = sample.entities
    relations = sample.relations

    for head, tail in zip(relations["head"], relations["tail"]):
        if (head, tail) not in relations_pairs:
            head_start = entities["start"][head]
            head_end = entities["end"][head]
            head_label = entities["label"][head]
            tail_start = entities["start"][tail]
            tail_end = entities["end"][tail]
            tail_label = entities["label"][tail]
            if head_label not in labels_to_exclude and tail_label not in labels_to_exclude:
                if head_start in words2input_ids.keys() and head_end in words2input_ids.keys() and \
                   tail_start in words2input_ids.keys() and tail_end in words2input_ids.keys():
                    head_start = words2input_ids[head_start][0]
                    head_end = words2input_ids[head_end][1]
                    head_label = label2id[head_label]
                    if not head in entitie2processed_entities.keys():
                        processed_entities["start"].append(head_start)
                        processed_entities["end"].append(head_end)
                        processed_entities["label"].append(head_label)
                        entitie2processed_entities[head] = len(
                            processed_entities["start"]) - 1
                    tail_start = words2input_ids[tail_start][0]
                    tail_end = words2input_ids[tail_end][1]
                    tail_label = label2id[tail_label]
                    if not tail in entitie2processed_entities.keys():
                        processed_entities["start"].append(tail_start)
                        processed_entities["end"].append(tail_end)
                        processed_entities["label"].append(tail_label)
                        entitie2processed_entities[tail] = len(
                            processed_entities["start"]) - 1
                    processed_relations["head"].append(
                        entitie2processed_entities[head])
                    processed_relations["tail"].append(
                        entitie2processed_entities[tail])
                    processed_relations["start_index"].append(head_start)
                    processed_relations["end_index"].append(tail_end)
                    relations_pairs.add((head, tail))

    return processed_entities, processed_relations


def process_sample(sample,
                   tokenizer,
                   id: str,
                   tc_label2id=None,
                   re_label2id=None,
                   max_lenght=MAX_LENGHT,
                   lowercase_all_words=False,
                   labels_to_exclude=None) -> EncodedDocumentSample:

    image = encode_image(sample.image)
    input_ids, bbox, labels, attention_mask, words2input_ids = process_words_boxes_labels(
        sample, tokenizer, tc_label2id, max_lenght, lowercase_all_words)
    entities, relations = process_entities_relations(sample, words2input_ids,
                                                     re_label2id,
                                                     labels_to_exclude)

    processed_sample = {
        "id": id,
        "input_ids": input_ids,
        "bbox": bbox,
        "labels": labels,
        "image": image,
        "entities": entities,
        "relations": relations,
        "attention_mask": attention_mask
    }

    processed_sample = EncodedDocumentSample(id=id,
                                             input_ids=input_ids,
                                             bbox=bbox,
                                             labels=labels,
                                             image=image,
                                             entities=entities,
                                             relations=relations,
                                             attention_mask=attention_mask)

    return processed_sample


def process_dataset(dataset: DocumentDataset,
                    splits: list[list[str]],
                    tokenizer,
                    labels_to_exclude=None,
                    lowercase_all_words=False) -> DocumentSamplesList:

    documents_samples_lists: list[DocumentSamplesList] = list()
    for split in splits:
        documents_samples_list = dataset
        for key in split:
            documents_samples_list = documents_samples_list[key]
        documents_samples_list = DocumentSamplesList(documents_samples_list)
        documents_samples_lists.append(documents_samples_list)

    samples_to_process = DocumentSamplesList()
    for documents_samples_list in documents_samples_lists:
        for sample in documents_samples_list:
            samples_to_process.append(sample)

    tc_label2id = dataset.labels["tokens"]["label2id"]
    re_label2id = dataset.labels["entities"]["label2id"]

    processed_dataset = DocumentSamplesList()
    with tqdm.tqdm(desc="Processing dataset",
                   total=len(samples_to_process)) as pbar:
        for sample in samples_to_process:
            processed_sample = process_sample(
                sample,
                tokenizer,
                id=sample.id,
                tc_label2id=tc_label2id,
                re_label2id=re_label2id,
                max_lenght=MAX_LENGHT,
                labels_to_exclude=labels_to_exclude,
                lowercase_all_words=lowercase_all_words)
            processed_dataset.append(processed_sample)
            pbar.update()

    return processed_dataset
