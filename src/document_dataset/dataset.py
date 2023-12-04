import PIL.Image
import numpy
import dataclasses
import typing
import pprint


class Sample:

    def __init__(self, id: str) -> None:
        self.id = id

    def __repr__(self):
        return f"DocumentSample {self.id}"

    def __getitem__(self, item):
        return self.__dict__[item]


class DocumentSample(Sample):

    def __init__(self,
                 id: str,
                 words: list[str],
                 boxes: list[list[int]],
                 labels: list[str],
                 entities: dict[str, list],
                 relations: dict[str, list],
                 image: typing.Union[None, PIL.Image.Image] = None) -> None:
        self.id = id
        self.words = words
        self.boxes = boxes
        self.labels = labels
        self.entities = entities
        self.relations = relations
        self.image = image


class EncodedDocumentSample(Sample):

    def __init__(self,
                 id: str,
                 input_ids: list[int],
                 bbox: list[list[int]],
                 labels: list[int],
                 entities: dict[str, list[int]],
                 relations: dict[str, list[int]],
                 attention_mask: list[int],
                 image: typing.Union[None, numpy.ndarray] = None) -> None:
        self.id = id
        self.input_ids = input_ids
        self.bbox = bbox
        self.labels = labels
        self.entities = entities
        self.relations = relations
        self.image = image
        self.attention_mask = attention_mask

    def __repr__(self):
        return f"DocumentSample {self.id}"

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()


class DocumentSamplesList(list[Sample]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.samples = self
        self.samples_map: dict[str, Sample] = dict()
        for sample in self:
            self.samples_map[sample.id] = sample

    def __repr__(self):
        return f"DocumentSamplesList:\n {super().__repr__()}"

    def __getitem__(self, item) -> Sample:
        if isinstance(item, str):
            return self.samples_map[item]
        return super().__getitem__(item)

    def append(self, __object: Sample) -> None:
        self.samples_map[__object.id] = __object
        return super().append(__object)

    def __add__(self, other):
        if isinstance(other, DocumentSamplesList):
            new = DocumentSamplesList(super().__add__(other))
            new.samples_map = {i: new[i] for i in range(len(new))}
            return new

    def extract_samples_labels(self):

        def create_id2label(labels: list[str]) -> dict[int, str]:
            id2label: dict[int, str] = {}
            for i in range(len(labels)):
                id2label[i] = labels[i]
            return id2label

        def create_label2id(labels: list[str]) -> dict[str, int]:
            label2id: dict[str, int] = {}
            for i in range(len(labels)):
                label2id[labels[i]] = i
            return label2id

        labels_tags = set()
        entities_labels_tags = set()
        prefixes = set()
        for sample in self.samples:
            for label in sample.entities["label"]:
                entities_labels_tags.add(label)
            for label in sample.labels:
                if label == "O":
                    continue
                prefix, label = label.split("-")
                prefixes.add(prefix)
                labels_tags.add(label)
        prefixes = sorted(list(prefixes))
        labels_tags = sorted(labels_tags)
        labels_tags = list(labels_tags)
        entities_labels = list(entities_labels_tags)
        tokens_labels = ["O"]
        for label in labels_tags:
            for prefix in prefixes:
                tokens_labels.append(prefix + "-" + label)

        return {
            "entities": {
                "labels": entities_labels,
                "id2label": create_id2label(entities_labels),
                "label2id": create_label2id(entities_labels)
            },
            "tokens": {
                "labels": tokens_labels,
                "id2label": create_id2label(tokens_labels),
                "label2id": create_label2id(tokens_labels)
            }
        }


@dataclasses.dataclass
class DocumentDataset:
    name: str
    samples: DocumentSamplesList
    splits: dict
    tag_format: str
    labels: dict
    citation: str = ""

    def __repr__(self):
        repr = f"Dataset name: {self.name}\n"
        repr += f"Tag format: {self.tag_format}\n"
        repr += f"Labels: {self.labels}\n"
        repr += pprint.pformat(self.splits, sort_dicts=False)
        return repr

    def __getitem__(self, item):
        return self.splits[item]
