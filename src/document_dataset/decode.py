import numpy
import PIL.Image

from .encode_decode import convert_chw_to_hwc, invert_color_channel


def decode_image(processed_image):
    processed_image = numpy.array(processed_image, "uint8")
    processed_image = convert_chw_to_hwc(processed_image)
    image = invert_color_channel(PIL.Image.fromarray(processed_image, "RGB"))
    return image


def decode_input_ids(input_ids, tokenizer):
    words = [
        tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids
    ]
    if len(input_ids) == len(words):
        return words
