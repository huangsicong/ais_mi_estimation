import torch
from ..registry import register


@register
class RecoverColorChannelLength(object):
    """
    Convert geryscaled rgb tensor back to 3 channels by repeating the greyscale channel 3 times.
    """

    def __init__(self, hparams):
        self.raw_rgb = hparams.dataset.input_dims[0] == 3
        self.raw_gray = hparams.dataset.input_dims[0] == 1

    def __call__(self, tensor):
        if self.raw_rgb:
            dims = tensor.shape
            if len(dims) == 3:  # C, H, W
                if dims[0] == 1:
                    return tensor.repeat([3, 1, 1])
                elif dims[0] == 3:
                    return tensor
                else:
                    raise ValueError("Tensor is not greyscale or rgb")
            elif len(dims) == 2:  # H, W
                return tensor.unsqueeze(0).repeat([3, 1, 1])

        elif self.raw_gray:
            dims = tensor.shape
            if len(dims) == 3:  # C, H, W
                if dims[0] == 1:
                    return tensor
                elif dims[0] == 3:
                    # assume the three channels were repeated, just return one of them
                    assert torch.equal(tensor[0], tensor[1]) & torch.equal(
                        tensor[0], tensor[2])
                    return tensor[0].unsqueeze(0)
                else:
                    raise ValueError("Tensor is not greyscale or rgb")

            elif len(dims) == 2:  # H, W
                return tensor.unsqueeze(0)

        return tensor
