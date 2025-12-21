from __future__ import annotations

import torch
import numpy as np
import math
import importlib
import random
from random import uniform
from torchvision.transforms import Compose
import time
GLOBAL_RANDOM_STATE = np.random.RandomState(47)
import torch.nn.functional as F
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor


def correct_crop_centers(
    centers: list[int],
    spatial_size: Sequence[int] | int,
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
):
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.

    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError(
                "The size of the proposed random crop ROI is larger than the image size, "
                f"got ROI size {spatial_size} and label image size {label_spatial_shape} respectively."
            )
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        center_i = min(max(c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return valid_centers  # type: ignore


def floor_divide(a: NdarrayOrTensor, b) -> NdarrayOrTensor:
    """`np.floor_divide` with equivalent implementation for torch.

    As of pt1.8, use `torch.div(..., rounding_mode="floor")`, and
    before that, use `torch.floor_divide`.

    Args:
        a: first array/tensor
        b: scalar to divide by

    Returns:
        Element-wise floor division between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor):
        return torch.floor_divide(a, b)
    return np.floor_divide(a, b)


def unravel_index(idx, shape) -> NdarrayOrTensor:
    """`np.unravel_index` with equivalent implementation for torch.

    Args:
        idx: index to unravel.
        shape: shape of array/tensor.

    Returns:
        Index unravelled for given shape
    """
    if isinstance(idx, torch.Tensor):
        coord = []
        for dim in reversed(shape):
            coord.append(idx % dim)
            idx = floor_divide(idx, dim)
        return torch.stack(coord[::-1])
    return np.asarray(np.unravel_index(idx, shape))


def nonzero(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.nonzero` with equivalent implementation for torch.

    Args:
        x: array/tensor.

    Returns:
        Index unravelled for given shape
    """
    if isinstance(x, np.ndarray):
        return np.nonzero(x)[0]
    return torch.nonzero(x).flatten()

def ravel(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.ravel` with equivalent implementation for torch.

    Args:
        x: array/tensor to ravel.

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "ravel"):  # `ravel` is new in torch 1.8.0
            return x.ravel()
        return x.flatten().contiguous()
    return np.ravel(x)


def any_np_pt(x: NdarrayOrTensor, axis: int | Sequence[int]) -> NdarrayOrTensor:
    """`np.any` with equivalent implementation for torch.

    For pytorch, convert to boolean for compatibility with older versions.

    Args:
        x: input array/tensor.
        axis: axis to perform `any` over.

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, np.ndarray):
        return np.any(x, axis)  # type: ignore

    # pytorch can't handle multiple dimensions to `any` so loop across them
    axis = [axis] if not isinstance(axis, Sequence) else axis
    for ax in axis:
        try:
            x = torch.any(x, ax)
        except RuntimeError:
            # older versions of pytorch require the input to be cast to boolean
            x = torch.any(x.bool(), ax)
    return x


class RandScaleIntensity:
    def __init__(self, factors, prob=0.1, dtype=np.float32):
        if isinstance(factors, (int, float)):
            self.factors = (-abs(factors), abs(factors))
        elif isinstance(factors, tuple) and len(factors) == 2:
            self.factors = (min(factors), max(factors))
        else:
            raise ValueError("Factors should be a number or pair of numbers.")

        self.prob = prob
        self.dtype = dtype
        self.factor = None  # This will be set randomly per call if needed

    def randomize(self):
        self.do_transform = random.random() < self.prob
        if self.do_transform:
            self.factor = uniform(*self.factors)

    def __call__(self, img, mask):
        if torch.is_tensor(img):
            img_t = img.float()
        else:
            raise TypeError("Input must be a numpy array or a PyTorch tensor.")

        self.randomize()

        if self.do_transform:
            img_t *= (1 + self.factor)

        return img_t, mask


class RandShiftIntensity:
    def __init__(self, offsets, prob=0.1, dtype=np.float32):
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        elif isinstance(offsets, tuple) and len(offsets) == 2:
            self.offsets = (min(offsets), max(offsets))
        else:
            raise ValueError("Factors should be a number or pair of numbers.")

        self.prob = prob
        self.dtype = dtype
        self.offset = None  # This will be set randomly per call if needed

    def randomize(self):
        self.do_transform = random.random() < self.prob
        if self.do_transform:
            self.offset = uniform(*self.offsets)

    def __call__(self, img, mask):
        if torch.is_tensor(img):
            img_t = img.float()
        else:
            raise TypeError("Input must be a numpy array or a PyTorch tensor.")

        self.randomize()

        if self.do_transform:
            img_t = img_t + self.offset

        return img_t, mask


class Compose_my(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    """
    Resizes the image to a given size. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    The resize operation is applied consistently to both raw and labeled datasets.
    """
    def __init__(self, size):
        """
        Parameters:
            size (tuple of ints): The target size for the images. Should be a tuple (D, H, W) or (C, D, H, W).
        """
        self.size = size

    def __call__(self, m, l):
        """
        Resize both the image and the label to the specified size.
        Parameters:
            m (torch.Tensor): The raw dataset image tensor.
            l (torch.Tensor): The labeled dataset image tensor.
        Returns:
            tuple: A tuple containing the resized image and label.
        """
        m = F.interpolate(m.float(), size=self.size, mode='bilinear', align_corners=False)
        l = F.interpolate(l, size=self.size, mode='nearest')

        return m.short(), l


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """
    def __init__(self, axis_prob=0.5, spatial_axis=0):
        self.axis_prob = axis_prob
        self.spatial_axis = spatial_axis

    def __call__(self, m, l):
        if random.random() < self.axis_prob:
            m = torch.flip(m, [self.spatial_axis])
            l = torch.flip(l, [self.spatial_axis])
        return m, l


class RandomCrop:
    def __init__(self, output_size, num_sample, random_state):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.num_sample = num_sample
        self.output_size = output_size

    def __call__(self, input, mask):
        # a = time.time()
        out_w, out_h, out_d = self.output_size[0], self.output_size[1], self.output_size[2]
        if input.shape[1] <= out_w or input.shape[2] <= out_h or input.shape[3] <= out_d:
            pw = max((out_w - input.shape[1]) // 2 + 3, 0)
            ph = max((out_h - input.shape[2]) // 2 + 3, 0)
            pd = max((out_d - input.shape[3]) // 2 + 3, 0)
            padding = (pd, pd, ph, ph, pw, pw)
            input = torch.nn.functional.pad(input, padding, 'constant', 0)
            mask = torch.nn.functional.pad(mask, padding, 'constant', 0)
        (_, w, h, d) = input.shape

        w_starts = self.random_state.randint(low=0, high=w - out_w + 1, size=self.num_sample)
        h_starts = self.random_state.randint(low=0, high=h - out_h + 1, size=self.num_sample)
        d_starts = self.random_state.randint(low=0, high=d - out_d + 1, size=self.num_sample)
        samples_input = []
        samples_mask = []

        # Crop patches
        for i in range(self.num_sample):
            w1, h1, d1 = w_starts[i].item(), h_starts[i].item(), d_starts[i].item()
            cropped_input = input[:, w1:w1 + out_w, h1:h1 + out_h, d1:d1 + out_d]
            cropped_mask = mask[:, w1:w1 + out_w, h1:h1 + out_h, d1:d1 + out_d]
            samples_input.append(cropped_input.unsqueeze(0))
            samples_mask.append(cropped_mask.unsqueeze(0))

        # Stack all samples together
        stacked_images = torch.cat(samples_input, dim=0)
        stacked_masks = torch.cat(samples_mask, dim=0)
        # b = time.time()
        # print('randomcrop', b - a)
        return stacked_images, stacked_masks


class RandomCrop_xy_ratio:
    def __init__(self, output_size, num_sample, random_state, pos_ratio, neg_ratio, image_threshold=0.0):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.num_sample = num_sample
        self.output_size = output_size
        self.image_threshold = image_threshold
        self.possibility = pos_ratio / (neg_ratio + pos_ratio)

    def __call__(self, input, mask):
        out_w, out_h, out_d = self.output_size[0], self.output_size[1], self.output_size[2]
        if input.shape[1] <= out_w or input.shape[2] <= out_h or input.shape[3] <= out_d:
            pw = max((out_w - input.shape[1]) // 2 + 3, 0)
            ph = max((out_h - input.shape[2]) // 2 + 3, 0)
            pd = max((out_d - input.shape[3]) // 2 + 3, 0)
            padding = (pd, pd, ph, ph, pw, pw)
            input = torch.nn.functional.pad(input, padding, 'constant', 0)
            mask = torch.nn.functional.pad(mask, padding, 'constant', 0)
        (_, w, h, d) = input.shape

        samples_input = []
        samples_mask = []

        if torch.any(mask) == 0:
            # a = time.time()
            h_starts = torch.randint(low=0, high=h - out_h + 1, size=(self.num_sample,))
            w_starts = torch.randint(low=0, high=w - out_w + 1, size=(self.num_sample,))
            d_starts = torch.randint(low=0, high=d - out_d + 1, size=(self.num_sample,))
            # Crop patches
            for i in range(self.num_sample):
                h1, w1, d1 = h_starts[i].item(), w_starts[i].item(), d_starts[i].item()
                cropped_input = input[:, w1:w1 + out_w, h1:h1 + out_h, d1:d1 + out_d]
                cropped_mask = mask[:, w1:w1 + out_w, h1:h1 + out_h, d1:d1 + out_d]
                samples_input.append(cropped_input.unsqueeze(0))
                samples_mask.append(cropped_mask.unsqueeze(0))
            # b = time.time()
        else:
            # a=time.time()
            # mask_any = torch.any(mask[1:], dim=0).bool()
            # fg_indices = torch.nonzero(mask_any)
            # input_any = torch.any(input >= self.image_threshold, dim=0).bool()
            # bg_indices = torch.nonzero(input_any & ~mask_any)
            label_flat = ravel(any_np_pt(mask[:], 0))  # in case label has multiple dimensions
            fg_indices = nonzero(label_flat)

            img_flat = ravel(any_np_pt(input > self.image_threshold, 0))
            bg_indices = nonzero(img_flat & ~label_flat)
            # random.seed(1)
            # np.random.seed(1)
            # torch.manual_seed(1)
            # torch.cuda.manual_seed(1)

            for k in range(self.num_sample):
                indices_to_use = fg_indices if random.uniform(0, 1) < self.possibility else bg_indices
                random_index = torch.randint(0, len(indices_to_use), (1,))

                random_non_zero_index = indices_to_use[random_index]

                center = unravel_index(random_non_zero_index, input.shape[1:]).tolist()
                center = [i[0] for i in center]
                center = correct_crop_centers(center, self.output_size, input.shape[1:], allow_smaller=False)
                # print(center)
                center_w, center_h, center_d = center[0], center[1], center[2]

                start_w = center_w - out_w // 2
                end_w = center_w + out_w // 2
                start_h = center_h - out_h // 2
                end_h = center_h + out_h // 2
                start_d = center_d - out_d // 2
                end_d = center_d + out_d // 2

                # start_w = max(0, center_w - out_w // 2)
                # start_h = max(0, center_h - out_h // 2)
                # start_d = max(0, center_d - out_d // 2)
                # end_w = min(w, start_w + out_w)
                # end_h = min(h, start_h + out_h)
                # end_d = min(d, start_d + out_d)
                # start_w = max(0, end_w - out_w)
                # start_h = max(0, end_h - out_h)  # 重新校正开始点，防止下界溢出# 重新校正开始点，防止左界溢出
                # start_d = max(0, end_d - out_d)

                cropped_mask = mask[:, start_w:end_w, start_h:end_h, start_d:end_d]
                cropped_input = input[:, start_w:end_w, start_h:end_h, start_d:end_d]

                samples_input.append(cropped_input.unsqueeze(0))
                samples_mask.append(cropped_mask.unsqueeze(0))
            # b = time.time()
            # print('randomcrop', b - a)

        # Stack all samples together
        stacked_images = torch.cat(samples_input, dim=0)
        stacked_masks = torch.cat(samples_mask, dim=0)

        return stacked_images, stacked_masks


class RandomCrop_xy:
    def __init__(self, output_size, num_sample, random_state):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.num_sample = num_sample
        self.output_size = output_size

    def __call__(self, input, mask):
        out_w, out_h, out_d = self.output_size[0], self.output_size[1], self.output_size[2]
        if input.shape[1] <= out_w or input.shape[2] <= out_h or input.shape[3] <= out_d:
            pw = max((out_w - input.shape[1]) // 2 + 3, 0)
            ph = max((out_h - input.shape[2]) // 2 + 3, 0)
            # pd = max((out_d - input.shape[3]) // 2 + 3, 0)
            padding = (0, 0, ph, ph, pw, pw)
            input = torch.nn.functional.pad(input, padding, 'constant', 0)
            mask = torch.nn.functional.pad(mask, padding, 'constant', 0)
        (_, w, h, d) = input.shape

        # h_starts = self.random_state.randint(low=0, high=h - out_h + 1, size=self.num_sample)
        # w_starts = self.random_state.randint(low=0, high=w - out_w + 1, size=self.num_sample)
        samples_input = []
        samples_mask = []

        h_starts = self.random_state.randint(low=0, high=h - out_h + 1, size=self.num_sample)
        w_starts = self.random_state.randint(low=0, high=w - out_w + 1, size=self.num_sample)
        # Crop patches
        for i in range(self.num_sample):
            h1, w1 = h_starts[i].item(), w_starts[i].item()
            cropped_input = input[:, w1:w1 + out_w, h1:h1 + out_h, :]
            cropped_mask = mask[:, w1:w1 + out_w, h1:h1 + out_h, :]
            samples_input.append(cropped_input.unsqueeze(0))
            samples_mask.append(cropped_mask.unsqueeze(0))


        # Stack all samples together
        stacked_images = torch.cat(samples_input, dim=0)
        stacked_masks = torch.cat(samples_mask, dim=0)

        return stacked_images, stacked_masks


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e., re-scaling the values
    to have 0-mean and 1-std deviation. This implementation is for PyTorch tensors.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=True, dim=(1,2,3)):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise
        self.dim = dim

    def __call__(self, m, l):
        # a = time.time()
        if self.mean is not None and self.std is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per channel
                mean = m.float().mean(dim=self.dim, keepdim=True)
                std = m.float().std(dim=self.dim, keepdim=True)
            else:
                mean = m.float().mean()
                std = m.float().std()

        std = torch.clamp(std, min=self.eps)
        m_normalized = (m - mean) / std
        return m_normalized, l


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor using PyTorch,
    i.e., shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value=None, max_value=None, clip_min=0, clip_max=1):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, m, l):
        if self.min_value is None:
            min_value = torch.min(m)
        else:
            min_value = self.min_value

        if self.max_value is None:
            max_value = torch.max(m)
        else:
            max_value = self.max_value

        m = (m - min_value) / (max_value - min_value)
        m = torch.clip(m, self.clip_min, self.clip_max)

        return m, l


class LabelMapping:
    """
    Apply simple min-max scaling to a given input tensor using PyTorch,
    i.e., shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, init_label=[1, [2,3,6,9], 4, [5,7], '7+'], new_label=[0,1,2,3,1]):
        self.init_label = init_label
        self.new_label = new_label
        assert len(init_label) == len(new_label)

    def __call__(self, m, l):
        l[l == 1] = 0
        l[(l == 2) | (l == 3) | (l >= 6)] = 1
        l[l == 4] = 2
        l[(l == 5)] = 3
        return m, l

class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m, l):
        # a = time.time()
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            # b = time.time()
            # print(b - a)
            return torch.clip(result, -1, 1), l
        # b = time.time()
        # print('randomcontrast:', b - a)
        return m, l


class RandomRotate90:
    """
    Rotate a tensor by 90 degrees around a randomly chosen plane. Tensor can be either 3D (DxHxW) or 4D (CxDxHxW).

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, execution_probability=0.1, type='2D', **kwargs):
        self.random_state = random_state
        # Always rotate around z-axis
        if type == '2D':
            self.axis = [-2, -1]
        elif type == '3D':
            self.axis = [-3, -2]
        self.execution_probability = execution_probability

    def __call__(self, m, l):
        assert m.ndim in [3, 4, 5]
        # a = time.time()
        if self.random_state.uniform() < self.execution_probability:
            # Pick number of rotations at random
            k = self.random_state.randint(0, 4)
            m = torch.rot90(m, k, self.axis)
            l = torch.rot90(l, k, self.axis)
        # b = time.time()
        # print('randomrotate:', b - a)
        return m, l


class Transformer:
    def __init__(self, phase_config):
        self.phase_config = phase_config
        self.seed = GLOBAL_RANDOM_STATE.randint(1000000)

    def image_transform(self):
        return self._create_transform('image')

    def label_transform(self):
        return self._create_transform('label')

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        transforms_instances = []
        for transform_info in self.phase_config[name]:
            # 获取变换的类名
            transform_name = transform_info['name']

            items = list(transform_info.items())[1:]  # 从第二个键值对开始的所有键值对
            transform_info = dict(items)

            if 'Random' in transform_name:
                transform_info['random_state'] = np.random.RandomState(self.seed)
            # 获取变换类
            transform_class = globals()[transform_name]
            # 创建变换实例，传递所有剩余的字典项作为关键字参数
            instance = transform_class(**transform_info)
            transforms_instances.append(instance)
        # 使用Compose创建一个组合变换
        return Compose(transforms_instances)


def test_StandardTransformer():
    config = {
        'image': [
            # {'name': 'RandomContrast', 'execution_probability': 0.5},
            {'name': 'RandomFlip'},
            {'name': 'Standardize'},
            # {'name': 'RandomRotate90'},
            # {'name': 'ToTensor', 'expand_dims': True}
        ],
        'label': [
            # {'name': 'RandomFlip', 'random_state':np.random.RandomState(47)},
            {'name': 'RandomFlip'},
            # {'name': 'RandomRotate90'},
            # {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
        ]
    }
    # base_config = {'mean': 0, 'std': 1}
    transformer = Transformer(config)
    raw_transforms = transformer.image_transform()
    a = torch.rand(10, 100, 100)
    b = raw_transforms(a)

    label_transforms = transformer.label_transform()
    c = torch.ones([10, 100, 100])
    d = label_transforms(c)


# test_StandardTransformer()