import tensorflow as tf

# split the image into patches
def split_image_into_patches(
    image, 
    patch_size: int=4) -> tf.Tensor:
    """
    split the image into patches sequence

    Parameters
    ----------
    image : tf.Tensor or array_like
        the image to split with shape `(B, H, W, C)`
    patch_size : int
        the patch size to split the image
    
    Returns
    -------
    patches : tf.Tensor
        the patches sequence with shape `(B, (H * W) / (patch_size^2), (patch_size^2) * C)`
    """
    # get the shape
    B, H, W, C = image.shape

    # patchs shape: (B, H // patch_size, W // patch_size, patch_size[0]*patch_size[1]*C)
    patchs = tf.image.extract_patches(
        image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    flat_size = (patch_size**2) * C
    patchs = tf.reshape(patchs, (B, -1, flat_size))

    return patchs