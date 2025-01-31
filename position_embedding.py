import cupy as cpy


def get_positional_embeddings(total_patches: int, dimension: int) -> cpy.ndarray:
    """Create positional embeddings.

    Args:
        total_patches: total number of pathces aka sequence length.
        dimension: dimension of the position embedding for a single patch.

    Returns:
        computed postion embedding.
    """
    position_embedding = cpy.zeros([total_patches, dimension])
    for pos in range(total_patches):
        for i in range(dimension):
            position_embedding[pos][i] = (
                cpy.sin(pos / (10000 ** (i / dimension)))
                if i % 2 == 0
                else cpy.cos(pos / (10000 ** ((i - 1) / dimension)))
            )
    return position_embedding
