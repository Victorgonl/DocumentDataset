import PIL.Image
import dataclasses
import typing
import pprint


@dataclasses.dataclass
class DocumentSample:
    id: str
    words: list[str]
    boxes: list[list[int]]
    labels: list[str]
    entities: dict[str, list[int]]
    relations: dict[str, list[int]]
    image: typing.Union[None, PIL.Image.Image]

    def __repr__(self):
        return self.id

    def __getitem__(self, item):
        return self.__dict__[item]


@dataclasses.dataclass
class DocumentDataset:
    name: str
    splits: dict
    tag_format: str
    tc_labels: list[str]
    tc_label2id: dict[str, int]
    tc_id2label: dict[int, str]
    re_labels: list[str]
    re_label2id: dict[str, int]
    re_id2label: dict[int, str]
    citation: str = ""

    def __repr__(self):
        repr = f"Dataset name: {self.name}"
        repr += f"Tag formar: {self.tag_format}"
        repr += f"Token Classification labels: {self.tc_labels}"
        repr += f"Relation Extraction labels: {self.re_labels}"
        repr += pprint.pformat(self.splits, sort_dicts=False)
        return repr

    def __getitem__(self, item):
        return self.splits[item]
