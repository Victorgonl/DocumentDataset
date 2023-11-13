import numpy
import PIL.Image

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BOX_NORMALIZER = 1000
CHANNEL_COUNT = 3


def invert_color_channel(image: PIL.Image.Image) -> PIL.Image.Image:
    image_array = numpy.array(image)
    image_array = image_array[:, :, ::-1]
    image = PIL.Image.fromarray(image_array, "RGB")
    return image


def convert_hwc_to_chw(image_array: numpy.ndarray) -> numpy.ndarray:
    return image_array.transpose(2, 0, 1)


def convert_chw_to_hwc(image_array: numpy.ndarray) -> numpy.ndarray:
    return image_array.transpose(1, 2, 0)


def resize_image(image: PIL.Image.Image) -> PIL.Image.Image:
    return image.resize((BOX_NORMALIZER, BOX_NORMALIZER))


def encode_image(image):
    image = image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = invert_color_channel(image)
    image_array = numpy.array(image)
    image_array = convert_hwc_to_chw(image_array)
    return image_array


def normalize_box(box, image_size):
    return [
        int(BOX_NORMALIZER * box[0] / image_size[0]),
        int(BOX_NORMALIZER * box[1] / image_size[1]),
        int(BOX_NORMALIZER * box[2] / image_size[0]),
        int(BOX_NORMALIZER * box[3] / image_size[1])
    ]


def normalize_boxes(boxes, image):
    image_size = image.size
    return [normalize_box(box, image_size) for box in boxes]


def unnormalize_box(box: list[int], image_size: tuple[int, int]) -> list[int]:
    return [
        int(image_size[0] * (box[0] / BOX_NORMALIZER)),
        int(image_size[1] * (box[1] / BOX_NORMALIZER)),
        int(image_size[0] * (box[2] / BOX_NORMALIZER)),
        int(image_size[1] * (box[3] / BOX_NORMALIZER))
    ]


def unnormalize_boxes(boxes: list[list[int]],
                      image: PIL.Image.Image) -> list[list[int]]:
    image_size = image.size
    return [unnormalize_box(box, image_size) for box in boxes]


def labels_to_ids(labels, label2id):
    return [label2id[label] for label in labels]


def ids_to_labels(labels, id2label):
    labels = labels.copy()
    for label in labels:
        try:
            label = id2label[label]
        except:
            pass
    return labels
