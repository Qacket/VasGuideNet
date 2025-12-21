import numpy as np
import os
import pickle
import zipfile
import lmdb
from multiprocessing import Pool
import SimpleITK as sitk
from monai.data.meta_tensor import MetaTensor
import torch
import torchvision
import logging
from monai import data, transforms
import argparse
from scipy.ndimage import zoom
import random


class LMDB_DatasetProcessor:
    def __init__(self, type, train_or_val, input_dir, output_dir, dir_name, mask_dir_name, npz_keys, spacing, num_phase,
                 liver_bound, logger=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dir_name = dir_name
        self.train_or_val = train_or_val
        self.mask_dir_name = mask_dir_name
        self.npz_keys = npz_keys
        self.logger = logger if logger else logging.getLogger(__name__)
        self.mkdir(self.output_dir)
        self.type = type
        self.spacing = spacing
        self.num_phase = num_phase
        self.liver_bound = liver_bound

    @staticmethod
    def encode_ct_data(ct_array):
        """将调整后的CT数据编码为两个uint8数组"""
        high_bytes = ct_array // 256
        low_bytes = ct_array % 256
        return high_bytes.astype(np.uint8), low_bytes.astype(np.uint8)

    @staticmethod
    def decode_ct_data(high_bytes, low_bytes):
        """从两个uint8数组解码CT数据"""
        reconstructed_mapped_values = high_bytes.astype(np.int16) * 256 + low_bytes.astype(np.int16)
        return reconstructed_mapped_values

    @staticmethod
    def mkdir(path):
        """Create directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def interpolate(volumeImage, newSpacing):
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)  ##此处为线性插值，其他插值方式可以去官网查询
        resampleFilter.SetOutputDirection(volumeImage.GetDirection())
        resampleFilter.SetOutputOrigin(volumeImage.GetOrigin())

        newSpacing = np.array(newSpacing, float)
        print("spacing before", volumeImage.GetSpacing())
        newSize = volumeImage.GetSize() / newSpacing * volumeImage.GetSpacing()
        newSize = newSize.astype(int)
        resampleFilter.SetSize(newSize.tolist())
        resampleFilter.SetOutputSpacing(newSpacing)
        newVolumeImage = resampleFilter.Execute(volumeImage)

        return newVolumeImage

    def process(self, input_dir, mask_dir, name, args, type, save_dir):
        try:
            slice_dict = {}  # 返回所有切片编码内容
            basename, ext = os.path.splitext(name)
            if ext == '.gz':
                # init_ct = sitk.ReadImage(os.path.join(input_dir, name))
                # init_mask = sitk.ReadImage(os.path.join(mask_dir, name))
                # spacing = init_ct.GetSpacing()
                # # 将 CT 和 mask 从原始分辨率重采样（Resample）到指定的 spacing（例如 1.5×1.5×2.0 mm³），
                # # 以便后续训练统一输入大小、尺度和空间信息。
                # if self.spacing != 0:
                #     init_ct = self.interpolate(init_ct, self.spacing)
                #     init_mask = self.interpolate(init_mask, self.spacing)
                #
                # ct = sitk.GetArrayFromImage(init_ct)
                # mask = sitk.GetArrayFromImage(init_mask)
                # ct = np.expand_dims(ct, axis=0)
                # mask = np.expand_dims(mask, axis=0)
                # c, d, h, w = mask.shape
                # # TODO shenchaojie add
                # c_init, d_init, h_init, w_init = mask.shape
                # name_save = name.replace('.nii.gz', '')

                # 修改gz重采样的方式 使用跟npz一样的通过monai来重采样
                # 读取CT和mask
                init_ct = sitk.ReadImage(os.path.join(input_dir, name))
                init_mask = sitk.ReadImage(os.path.join(mask_dir, name))
                spacing = init_ct.GetSpacing()  # (z, y, x)
                # 转为 numpy [C, D, H, W]
                ct = sitk.GetArrayFromImage(init_ct)  # [D, H, W]
                mask = sitk.GetArrayFromImage(init_mask)  # [D, H, W]
                ct = np.expand_dims(ct, axis=0)
                mask = np.expand_dims(mask, axis=0)

                c_init, d_init, h_init, w_init = mask.shape

                if self.spacing != 0:
                    # SimpleITK spacing: (z, y, x) → 转为 MONAI 需要的 (x, y, z)
                    original_spacing = [spacing[2], spacing[1], spacing[0]]
                    print('original_spacing', original_spacing)
                    # 构造仿射矩阵
                    affine_matrix = np.eye(4)
                    affine_matrix[0, 0] = original_spacing[0]  # spacing_x
                    affine_matrix[1, 1] = original_spacing[1]  # spacing_y
                    affine_matrix[2, 2] = original_spacing[2]  # spacing_z

                    if type == 'train':
                        data_dict = {
                            "image": MetaTensor(ct, affine=affine_matrix),
                            "label": MetaTensor(mask, affine=affine_matrix)
                        }

                        spacing_transform = transforms.Compose([
                            transforms.Spacingd(
                                keys=["image", "label"],
                                pixdim=(self.spacing[2], self.spacing[1], self.spacing[0]),
                                mode=("bilinear", "nearest")
                            )
                        ])
                        data_dict = spacing_transform(data_dict)
                        ct = np.array(data_dict["image"])
                        mask = np.array(data_dict["label"])
                    else:
                        data_dict = {
                            "image": MetaTensor(ct, affine=affine_matrix)
                        }

                        spacing_transform = transforms.Compose([
                            transforms.Spacingd(
                                keys=["image"],
                                pixdim=(self.spacing[2], self.spacing[1], self.spacing[0]),
                                mode=("bilinear",)
                            )
                        ])
                        data_dict = spacing_transform(data_dict)
                        ct = np.array(data_dict["image"])

                name_save = name.replace('.nii.gz', '')

            elif ext == '.npz':
                data = np.load(os.path.join(input_dir, name), allow_pickle=True)
                ct, mask, spacing, liver_mask = (data[self.npz_keys[0]][self.num_phase],
                                                 data[self.npz_keys[1]][self.num_phase],
                                                 data[self.npz_keys[2]][self.num_phase],
                                                 data[self.npz_keys[3]][self.num_phase])
                high = mask.shape[1]
                print(os.path.join(input_dir, name))
                if self.liver_bound != 0:
                    s_z = np.where(liver_mask > 0)[1]
                    if len(s_z) > 0:
                        min_z = min(s_z)
                        max_z = max(s_z)
                        ct = ct[:, max(min_z - self.liver_bound, 0):min(max_z + self.liver_bound, high), :, :]
                        mask = mask[:, max(min_z - self.liver_bound, 0):min(max_z + self.liver_bound, high), :, :]
                        liver_mask = liver_mask[:, max(min_z - self.liver_bound, 0):min(max_z + self.liver_bound, high),
                                     :, :]
                    else:
                        print('no liver mask in {name}')

                c_init, d_init, h_init, w_init = mask.shape
                if self.spacing != 0:
                    assert np.all(spacing == spacing[0]), "spacing is not equal"
                    if type == 'train':
                        original_spacing = [spacing[0][2], spacing[0][1], spacing[0][0]]  # 替换成实际的spacing值
                        # 创建一个仿射矩阵，这通常是一个 4x4 的单位矩阵，对角线替换为 spacing 和 1
                        affine_matrix = np.eye(4)
                        affine_matrix[0, 0] = original_spacing[0]  # spacing_x
                        affine_matrix[1, 1] = original_spacing[1]  # spacing_y
                        affine_matrix[2, 2] = original_spacing[2]  # spacing_z
                        dict = {}
                        dict['image'] = MetaTensor(ct, affine=affine_matrix)
                        dict['label'] = MetaTensor(mask, affine=affine_matrix)
                        dict['liver_mask'] = MetaTensor(liver_mask, affine=affine_matrix)
                        spacing_transform = transforms.Compose(
                            [transforms.Spacingd(keys=["image", "label", "liver_mask"],
                                                 pixdim=(self.spacing[2], self.spacing[1], self.spacing[0]),
                                                 mode=("bilinear", "nearest", "nearest")
                                                 )])
                        a = spacing_transform(dict)
                        ct_spacing = np.array(a['image'])
                        mask_spacing = np.array(a['label'])
                        liver_mask_spacing = np.array(a['liver_mask'])

                        ct = ct_spacing
                        mask = mask_spacing
                        liver_mask = liver_mask_spacing
                    else:
                        original_spacing = [spacing[0][2], spacing[0][1], spacing[0][0]]
                        affine_matrix = np.eye(4)
                        affine_matrix[0, 0] = original_spacing[0]  # spacing_x
                        affine_matrix[1, 1] = original_spacing[1]  # spacing_y
                        affine_matrix[2, 2] = original_spacing[2]  # spacing_z
                        dict = {}
                        dict['image'] = MetaTensor(ct, affine=affine_matrix)
                        spacing_transform = transforms.Compose(
                            [transforms.Spacingd(keys=["image"],
                                                 pixdim=(self.spacing[2], self.spacing[1], self.spacing[0]),
                                                 mode=("bilinear"))])
                        a = spacing_transform(dict)
                        ct_spacing = np.array(a['image'])
                        ct = ct_spacing

                name_save = name.replace('.npz', '')

            ct_min = ct.min()
            if ct_min < 0:
                ct = ct - ct_min

            c, d, h, w = ct.shape

            # env = lmdb.open(save_dir, map_size=1099511627776 * 2)
            # if type == 'train':
            #     with env.begin(write=True) as txn:
            #         for idx in range(d):
            #             ct_i = ct[:, idx]
            #             mask_i = mask[:, idx]
            #             high_bytes, low_bytes = self.encode_ct_data(ct_i)
            #             high_bytes = torch.from_numpy(high_bytes)
            #             low_bytes = torch.from_numpy(low_bytes)
            #             mask_i = torch.from_numpy(mask_i.astype(np.uint8))
            #             data = torch.stack([high_bytes, low_bytes, mask_i], 0)
            #             data = torchvision.io.encode_png(data.reshape(1, -1, w))
            #             bin_data = pickle.dumps(data)
            #             txn.put(str(name_save + f"_{idx}").encode(), bin_data)
            #
            #     env.close()
            #     # self.logger.info(os.path.join(input_dir, name) + " done~")
            #     # print(os.path.join(input_dir, name) + " done~")
            #     print(mask.shape)
            #     return name_save, ct_min, ct.shape, [c_init, w_init, h_init, d_init, spacing[0]]
            # else:
            #     with env.begin(write=True) as txn:
            #         for idx in range(d):
            #             ct_i = ct[:, idx]
            #             high_bytes, low_bytes = self.encode_ct_data(ct_i)
            #             high_bytes = torch.from_numpy(high_bytes)
            #             low_bytes = torch.from_numpy(low_bytes)
            #             data1 = torch.stack([high_bytes, low_bytes], 0)
            #             data1 = torchvision.io.encode_png(data1.reshape(1, -1, w))
            #             bin_data1 = pickle.dumps(data1)
            #             txn.put(str(name_save + "_image" + f"_{idx}").encode(), bin_data1)
            #
            #         for idx in range(d_init):
            #             mask_i = mask[:, idx]
            #             mask_i = torch.from_numpy(mask_i.astype(np.uint8))
            #             data2 = torchvision.io.encode_png(mask_i.reshape(1, -1, w_init))
            #             bin_data2 = pickle.dumps(data2)
            #             txn.put(str(name_save + "_label" + f"_{idx}").encode(), bin_data2)
            #
            #     env.close()
            #     # self.logger.info(os.path.join(input_dir, name) + " done~")
            #     # print(os.path.join(input_dir, name) + " done~")
            #     print(mask.shape)
            #     return name_save, ct_min, ct.shape, [c_init, w_init, h_init, d_init, spacing[0]]
            if type == 'train':
                for idx in range(d):
                    ct_i = ct[:, idx]
                    mask_i = mask[:, idx]
                    high_bytes, low_bytes = self.encode_ct_data(ct_i)
                    high_bytes = torch.from_numpy(high_bytes)
                    low_bytes = torch.from_numpy(low_bytes)
                    mask_i = torch.from_numpy(mask_i.astype(np.uint8))
                    data = torch.stack([high_bytes, low_bytes, mask_i], 0)
                    data = torchvision.io.encode_png(data.reshape(1, -1, w))
                    bin_data = pickle.dumps(data)
                    slice_dict[str(name_save + f"_{idx}").encode()] = bin_data

                return name_save, ct_min, ct.shape, [c_init, w_init, h_init, d_init, spacing[0]], slice_dict
            else:
                for idx in range(d):
                    ct_i = ct[:, idx]
                    high_bytes, low_bytes = self.encode_ct_data(ct_i)
                    high_bytes = torch.from_numpy(high_bytes)
                    low_bytes = torch.from_numpy(low_bytes)
                    data1 = torch.stack([high_bytes, low_bytes], 0)
                    data1 = torchvision.io.encode_png(data1.reshape(1, -1, w))
                    bin_data1 = pickle.dumps(data1)
                    slice_dict[str(name_save + "_image" + f"_{idx}").encode()] = bin_data1

                for idx in range(d_init):
                    mask_i = mask[:, idx]
                    mask_i = torch.from_numpy(mask_i.astype(np.uint8))
                    data2 = torchvision.io.encode_png(mask_i.reshape(1, -1, w_init))
                    bin_data2 = pickle.dumps(data2)
                    slice_dict[str(name_save + "_label" + f"_{idx}").encode()] = bin_data2

                print(mask.shape)
                return name_save, ct_min, ct.shape, [c_init, w_init, h_init, d_init, spacing[0]], slice_dict
        except zipfile.BadZipfile:
            self.logger.error(f"{name} is a bad zipfile.")
            return None

    def read_file_list(self):

        args_list_total = []
        for input_dir_index, item in enumerate(self.input_dir):
            input_dir = self.input_dir[input_dir_index]

            self.logger.info("The root path of datasets is %s" % input_dir)

            for dir_index, item in enumerate(self.dir_name):
                # 获取对应的 mask_dir_name
                mask_dir = self.mask_dir_name[dir_index]
                for dir_i in os.listdir(input_dir):
                    # 如果当前项是你感兴趣的目录
                    if dir_i == item:
                        # TODO 这里不要了
                        # # 打开之前生成的 LMDB 数据库。
                        # # 读取 key 为 b'__keys__' 的内容（这个 key 用来保存所有图像数据索引信息）。
                        # # pickle.loads() 将其反序列化成 Python 对象（通常是个 dict）。
                        # # 将这些信息合并（更新）到 keys 中，避免重复处理数据。
                        # env = lmdb.open(self.output_dir, map_size=1099511627776 * 2)
                        # with env.begin(write=False) as txn:
                        #     _keys = txn.get(b'__keys__')
                        #     print(_keys)
                        #     if _keys is not None:
                        #         keys.update(pickle.loads(_keys))
                        # env.close()
                        # info = []
                        # for k, v in keys.items():
                        #     info.extend(v)
                        #
                        # info_dict_ready = [(key, (value1, tuple_value, list_)) for key, value1, tuple_value, list_ in info]
                        # 现在转换为字典
                        # info = dict(info_dict_ready)
                        self.logger.info(f"Start process dataset {input_dir}")

                        if self.type == 'npz':
                            # src_files = set(os.listdir('/data4/liver_CT4_Z2_tr5.2/'))
                            tmp_args_list = [(os.path.join(input_dir, dir_i), 0, i, self) for i in
                                             os.listdir(os.path.join(input_dir, dir_i))]
                        elif self.type == 'nii.gz':
                            tmp_args_list = [
                                (os.path.join(input_dir, dir_i), os.path.join(input_dir, mask_dir), i, self) for i
                                in
                                os.listdir(os.path.join(input_dir, dir_i))]

                        args_list_total.extend(tmp_args_list)

        return args_list_total

    def save_lmdb(self, args_list, type, save_dir):
        keys = {}
        os.makedirs(save_dir, exist_ok=True)
        print("type is %s" % type)
        print("save_dir is %s" % save_dir)

        new_args_list = [(item[0], item[1], item[2], item[3], type, save_dir) for item in args_list]

        pool = Pool(20)
        ret = pool.starmap(self.process, new_args_list)
        pool.close()
        pool.join()

        ret = [i for i in ret if i is not None]

        env = lmdb.open(save_dir, map_size=1099511627776 * 2)
        keys_tmp = []
        with env.begin(write=True) as txn:
            for name, ct_min, shape, result_list, slice_dict in ret:
                keys_tmp.append((name, ct_min, shape, result_list))
                for key, value in slice_dict.items():
                    txn.put(key, value)
        keys[type] = keys_tmp
        txn = env.begin(write=True)
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.commit()
        env.close()

    def running_npz(self):
        args_list_total = self.read_file_list()

        # 打乱数据，保证随机性
        random.shuffle(args_list_total)

        # 按比例划分 8:1:1 (训练:验证:测试)

        total_num = len(args_list_total)
        train_end = int(total_num * 0.8)
        val_end = train_end + int(total_num * 0.1)

        train_list = args_list_total[:train_end]
        val_list = args_list_total[train_end:val_end]
        test_list = args_list_total[val_end:]

        print("total num is %d" % total_num)
        print("train num is %d" % len(train_list))
        print("val num is %d" % len(val_list))
        print("test num is %d" % len(test_list))
        self.save_lmdb(train_list, 'train', os.path.join(self.output_dir, 'train'))
        self.save_lmdb(val_list, 'val', os.path.join(self.output_dir, 'val'))
        self.save_lmdb(test_list, 'test', os.path.join(self.output_dir, 'test'))

    def get_total_size(self):
        num = 0
        for dir_index, item in enumerate(self.dir_name):
            for dir_i in os.listdir(self.input_dir):
                if dir_i == item:
                    num += len(os.listdir(os.path.join(self.input_dir, dir_i)))
        return num


def create_lmdb_entry_point():
    parser = argparse.ArgumentParser('SingleSliceMaskRetinaNet')
    parser.add_argument('--type', type=str, default='nii.gz')  # npz的话是[phase, z, x, y]   nii的话是[x, y, z]
    parser.add_argument('--npz_keys', type=str, nargs='+',
                        default=['ct', 'tumor_mask', 'spacing', 'liver_mask'])  # 当type为nii.gz时，不用管这一项，因为nii.gz文件没有keys
    parser.add_argument('--train_or_val', type=str,
                        default='train')  # 造训练数据需要image和label同时resample，val只需要对imageresample
    parser.add_argument('--input_dir', type=str, nargs='+',
                        default=['/data0/scj/datasets/肝八段数据集/Vessel/PuJian_20241109/',
                                 '/data0/scj/datasets/肝八段数据集/Vessel/abdomen_3d_reconstruction/'])

    # parser.add_argument('--dir_name', type=str, nargs='+',
    #                     default=['20230410_zheer', '20231227_zheer','20240105_zheerv1','20240105_zheerv2','20240110_zheer','20240112_zheer',
    #                              '20240118_zheer','20240122_zheer','20240826_yongkang_danqi','20240909_yongkang_danqi',
    #                              '20241227_yongkang','20240911_yongkang_danqi','20241230_yongkang_danqi','20250108_yongkang_danqi',
    #                              '20250311_yongkang_danqi','liver_CT4_Z2_tr5.2'])
    # parser.add_argument('--mask_dir_name', type=str, nargs='+',
    #                     default=['20230410_zheer', '20231227_zheer','20240105_zheerv1','20240105_zheerv2','20240110_zheer','20240112_zheer',
    #                              '20240118_zheer','20240122_zheer','20240826_yongkang_danqi','20240909_yongkang_danqi',
    #                              '20241227_yongkang','20240911_yongkang_danqi','20241230_yongkang_danqi','20250108_yongkang_danqi',
    #                              '20250311_yongkang_danqi','liver_CT4_Z2_tr5.2'])

    parser.add_argument('--dir_name', type=str, nargs='+', default=['imagesTr'])
    parser.add_argument('--mask_dir_name', type=str, nargs='+', default=['labelsTr'])

    # parser.add_argument('--dir_name', type=str, nargs='+', default=[
    # '20230227_guangxi','20230227_ningxia','20230227_shenzhen','20230320_smalltumor','20230330_guangxi','20230330_ningxia',
    # '20230330_shenzhen','20230410_guangxi','20230410_xiangya','20230410_zheer','20230526_ningxia','20230526_shenzhen',
    # '20230823_guangxi','20230823_shenzhen','20230823_xiangya','20231227_zheer','20240105_zheerv1','20240105_zheerv2',
    # '20240110_zheer','20240112_zheer','20240118_zheer','20240122_zheer','20240826_yongkang_danqi','20240909_yongkang_danqi',
    # '20241227_yongkang','20240911_yongkang_danqi','20241230_yongkang_danqi','20250108_yongkang_danqi','20250311_yongkang_danqi',
    # 'liver_CT4_Z2_tr5.2'])
    # parser.add_argument('--mask_dir_name', type=str, nargs='+', default=[
    # '20230227_guangxi','20230227_ningxia','20230227_shenzhen','20230320_smalltumor','20230330_guangxi','20230330_ningxia',
    # '20230330_shenzhen','20230410_guangxi','20230410_xiangya','20230410_zheer','20230526_ningxia','20230526_shenzhen',
    # '20230823_guangxi','20230823_shenzhen','20230823_xiangya','20231227_zheer','20240105_zheerv1','20240105_zheerv2',
    # '20240110_zheer','20240112_zheer','20240118_zheer','20240122_zheer','20240826_yongkang_danqi','20240909_yongkang_danqi',
    # '20241227_yongkang','20240911_yongkang_danqi','20241230_yongkang_danqi','20250108_yongkang_danqi','20250311_yongkang_danqi',
    # 'liver_CT4_Z2_tr5.2'])
    parser.add_argument('--output_dir', type=str, default="/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private")
    parser.add_argument('--spacing', nargs='+', type=int, default=[1.5, 1.5, 2.0])
    parser.add_argument('--num_phase', type=int, default=[0])
    parser.add_argument('--liver_bound', type=int, default=5)
    args = parser.parse_args()

    processor = LMDB_DatasetProcessor(
        type=args.type,
        train_or_val=args.train_or_val,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dir_name=args.dir_name,  # 可以是单个名称或名称列表
        mask_dir_name=args.mask_dir_name,
        npz_keys=args.npz_keys,
        spacing=args.spacing,
        num_phase=args.num_phase,
        liver_bound=args.liver_bound
    )

    processor.running_npz()


def read_lmdb_test():
    import pandas
    import io

    def decode_ct_data(high_bytes, low_bytes: torch.Tensor):
        """从两个uint8数组解码CT数据"""
        # 重构映射后的值
        high_bytes_int16 = high_bytes.to(dtype=torch.int16)
        low_bytes_int16 = low_bytes.to(dtype=torch.int16)
        reconstructed_mapped_values = high_bytes_int16 * 256 + low_bytes_int16
        return reconstructed_mapped_values

    def save_niigz(img, mask, save_dir):
        # 保存路径
        os.makedirs(save_dir, exist_ok=True)
        ct_save_path = os.path.join(save_dir, f"{case_id}_ct.nii.gz")
        mask_save_path = os.path.join(save_dir, f"{case_id}_mask.nii.gz")

        spacing = (1.5, 1.5, 2.0)
        image_array = img[0].cpu().numpy()  # shape: [D, H, W]
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing((spacing[2], spacing[1], spacing[0]))  # 注意：SimpleITK 是 (x, y, z)
        sitk.WriteImage(image, ct_save_path)

        mask_array = mask[0].cpu().numpy()  # shape: [D, H, W]
        mask = sitk.GetImageFromArray(mask_array)
        mask.SetSpacing((spacing[2], spacing[1], spacing[0]))  # 注意：SimpleITK 是 (x, y, z)
        sitk.WriteImage(mask, mask_save_path)

    input_dir = "/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private/train"
    env = lmdb.open(input_dir, readonly=True, lock=False,
                    readahead=False, meminit=False, create=False)
    filename_list = []
    info = []
    with env.begin(write=False) as txn:
        for k, v in pickle.loads(txn.get(b'__keys__')).items():
            for i in v:
                if i[0] in filename_list:
                    continue
                else:
                    info.extend([i])
    env.close()
    print(info)
    info_name = {}
    new_list = []
    num_multiphase = 1
    for ct_name, value, (_, max_depth, width, height), result_list in info:
        info_name[ct_name] = (value, num_multiphase, (0, max_depth, width, height), result_list)
        new_list.append(ct_name)
    print(info_name)
    print(new_list)
    case_id = 'PuJian_20241109_079_0001'
    ct_min, num_multiphase, i, shape = info_name[case_id]
    print(ct_min, num_multiphase, i, shape)
    env = lmdb.open(input_dir, readonly=True, lock=False,
                    readahead=False, meminit=False, create=False)
    txn = env.begin(buffers=True)

    img_all = []
    mask_all = []

    for t in range(i[0], i[1]):
        bin_data = txn.get(str(case_id + f"_{t}").encode())
        data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
        high_bytes, low_bytes, mask = torch.reshape(data, (3, -1, i[-2], i[-1]))[:, :num_multiphase, :, :]
        ct = decode_ct_data(high_bytes, low_bytes)
        ct = ct + ct_min
        img_all.append(ct)
        mask_all.append(mask)
    img = torch.stack(img_all, 1)
    mask = torch.stack(mask_all, 1)
    print(img.shape)  # c,d,h,w    (1, 234, 232, 232 )
    print(mask.shape)  # c,d,h,w

    save_dir = input_dir
    save_niigz(img, mask, save_dir)

    out_phase_id = 0
    image = img.permute(0, 3, 2, 1).contiguous()[0]
    mask = mask.permute(0, 3, 2, 1).contiguous()[0]
    print(image.shape)  # w h d
    print(mask.shape)  # w h d


if __name__ == '__main__':
    # create_lmdb_entry_point()
    read_lmdb_test()
