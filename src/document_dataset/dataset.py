import PIL.Image
import dataclasses
import typing
import pprint


@dataclasses.dataclass
class DocInfExtSample:
    id: str
    words: list[str]
    boxes: list[list[int]]
    labels: list[str]
    entities: dict[str, list[int]]
    relations: dict[str, list[int]]
    image: typing.Union[None, PIL.Image.Image]

    def __repr__(self):
        return self.id


@dataclasses.dataclass
class DocInfExtDataset:
    name: str
    samples: dict[str, DocInfExtSample]
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
        return pprint.pformat(self.splits, sort_dicts=False)

    def __getitem__(self, item):
        return self.splits[item]
