import io

from PIL import Image
from PIL.Image import Image as PILImage

from .session_factory import new_session

def get_concat_v_multi(imgs) -> PILImage:
    """
    Concatenate multiple images vertically.

    Args:
        imgs (List[PILImage]): The list of images to be concatenated.

    Returns:
        PILImage: The concatenated image.
    """
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    """
    Concatenate two images vertically.

    Args:
        img1 (PILImage): The first image.
        img2 (PILImage): The second image to be concatenated below the first image.

    Returns:
        PILImage: The concatenated image.
    """
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst



def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    """
    Perform a simple cutout operation on an image using a mask.

    This function takes a PIL image `img` and a PIL image `mask` as input.
    It uses the mask to create a new image where the pixels from `img` are
    cut out based on the mask.

    The function returns a PIL image representing the cutout of the original
    image using the mask.
    """
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

def remove(data):
    img = Image.open(io.BytesIO(data))

    session = new_session("u2net")

    masks = session.predict(img)

    cutouts = []
    for mask in masks:
        cutout = naive_cutout(img, mask)
        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()