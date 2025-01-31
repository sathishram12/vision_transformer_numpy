import cupy as cpy


def convert_image_to_patches(images: cpy.ndarray, num_patches_1d: int) -> cpy.ndarray:
    """Convert image into patches.

    Args:
        images: input images.
        num_patches_1d: total number of patches in one direction, same is replicated in other.

    Returns:
        images converted to patches.
    """
    n, c, h, w = images.shape
    patches = cpy.zeros([n, num_patches_1d**2, h * w * c // (num_patches_1d**2)])
    patch_size = h // num_patches_1d
    for index, image in enumerate(images):
        for i in range(num_patches_1d):
            for j in range(num_patches_1d):
                patch = image[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
                patches[index, i * num_patches_1d + j] = patch.flatten()
    return patches
