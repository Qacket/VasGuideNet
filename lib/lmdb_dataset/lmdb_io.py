import io
import pickle
import lmdb
import pandas
from torch.utils.data.dataset import Dataset
import torchvision
import torch


def decode_ct_data(high_bytes, low_bytes: torch.Tensor):
    """从两个uint8数组解码CT数据"""
    # 重构映射后的值
    high_bytes_int16 = high_bytes.to(dtype=torch.int16)
    low_bytes_int16 = low_bytes.to(dtype=torch.int16)
    reconstructed_mapped_values = high_bytes_int16 * 256 + low_bytes_int16
    return reconstructed_mapped_values


class LMDBDataset(Dataset):
    def __init__(self, input_dir, dataset_type, num_multiphase, type) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.dataset_type = dataset_type
        self.num_multiphase = num_multiphase
        self.type = type
        self.info_name = {}
        env = lmdb.open(self.input_dir, readonly=True, lock=False,
                        readahead=False, meminit=False, create=False)
        filename_list = []
        self.info = []
        with env.begin(write=False) as txn:
            for k, v in pickle.loads(txn.get(b'__keys__')).items():
                for i in v:
                    if i[0] in filename_list:
                        continue
                    else:
                        self.info.extend([i])
        env.close()
        if self.type == '3D':
            self.all_total_data_list = self.flatten_data(self.info)
        elif self.type == '2D':
            self.all_total_data_list = self.flatten_data_2D(self.info)

    def flatten_data(self, data):
        new_list = []
        for ct_name, value, (_, max_depth, width, height), result_list in data:
            self.info_name[ct_name] = (value, self.num_multiphase, (0, max_depth, width, height), result_list)
            new_list.append(ct_name)
        return new_list

    def flatten_data_2D(self, data):
        new_list = []
        for ct_name, value, (_, max_depth, width, height), result_list in data:
            for i in range(max_depth):
                self.info_name[ct_name + '-index' + str(i)] = (value, self.num_multiphase, (0, i, width, height),
                                                               result_list)
                new_list.append(ct_name + '-index' + str(i))
        return new_list

    def open_lmdb(self):
        self.env = lmdb.open(self.input_dir, readonly=True, lock=False,
                             readahead=False, meminit=False, create=False)
        self.txn = self.env.begin(buffers=True)

    def read_slices_train(self, case_id):
        ct_min, num_multiphase, i, shape = self.info_name[case_id]
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        img_all = []
        mask_all = []
        for t in range(i[0], i[1]):
            bin_data = self.txn.get(str(case_id + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            high_bytes, low_bytes, mask = torch.reshape(data, (3, -1, i[-2], i[-1]))[:, :num_multiphase, :, :]
            ct = decode_ct_data(high_bytes, low_bytes)
            ct = ct + ct_min
            img_all.append(ct)
            mask_all.append(mask)
        img = torch.stack(img_all, 1)
        mask = torch.stack(mask_all, 1)
        return img, mask

    def read_slices_val(self, case_id):
        ct_min, num_multiphase, i, shape = self.info_name[case_id]
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        img_all = []
        mask_all = []
        for t in range(i[0], i[1]):
            bin_data = self.txn.get(str(case_id + "_image" + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            high_bytes, low_bytes = torch.reshape(data, (2, -1, i[-2], i[-1]))[:, :num_multiphase, :, :]
            ct = decode_ct_data(high_bytes, low_bytes)
            ct = ct + ct_min
            img_all.append(ct)
        img = torch.stack(img_all, 1)

        for t in range(0, shape[3]):
            bin_data = self.txn.get(str(case_id + "_label" + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            mask = torch.reshape(data, (-1, shape[1], shape[2]))[:num_multiphase, :, :]
            mask_all.append(mask)
        mask = torch.stack(mask_all, 1)
        return img, mask, shape


class Dataset_3D(LMDBDataset):
    def __init__(self,
                 input_dir,
                 num_multiphase=1,
                 out_phase_id=[0, 1, 2, 3],
                 dataset_type='train',
                 transforms=None,
                 type='3D'):
        super().__init__(input_dir, dataset_type, num_multiphase, type)
        self.type = type
        self.info = []
        self.input_dir = input_dir
        self.num_multiphase = num_multiphase
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.out_phase_id = out_phase_id

    def __len__(self):
        return len(self.all_total_data_list)

    def __getitem__(self, index):
        name = self.all_total_data_list[index]
        if self.dataset_type == 'train':
            image, mask = self.read_slices_train(name)

            image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            if self.transforms != None:
                image, mask = self.transforms(image, mask)
            return image, mask

        elif self.dataset_type == 'val':
            image, mask, shape = self.read_slices_val(name)
            if len(image) != self.num_multiphase:
                return image, mask, shape, name
            else:
                image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                if self.transforms != None:
                    image, mask = self.transforms(image, mask)
                return image, mask, shape, name


class Dataset_3D_my(LMDBDataset):
    def __init__(self,
                 input_dir,
                 num_multiphase=1,
                 out_phase_id=[0, 1, 2, 3],
                 dataset_type='train',
                 transforms=None,
                 type='3D'):
        super().__init__(input_dir, dataset_type, num_multiphase, type)
        self.type = type
        self.info = []
        self.input_dir = input_dir
        self.num_multiphase = num_multiphase
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.out_phase_id = out_phase_id

    def __len__(self):
        return len(self.all_total_data_list)

    def __getitem__(self, index):
        name = self.all_total_data_list[index]
        if self.dataset_type == 'train':
            image, mask = self.read_slices_train(name)

            image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            sample = {"image": image, "label": mask}
            if self.transforms != None:
                sample = self.transforms(sample)
            return sample

        elif self.dataset_type == 'val':
            image, mask, shape = self.read_slices_val(name)
            if len(image) != self.num_multiphase:
                return {"image": image, "label": mask}
            else:
                image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                sample = {"image": image, "label": mask}
                if self.transforms != None:
                    sample = self.transforms(sample)
                return sample


class Dataset_3D_my_couinaud(Dataset):
    def __init__(self,
                 input_dir,
                 num_multiphase=1,
                 out_phase_id=[0, 1, 2, 3],
                 dataset_type='train',
                 transforms=None,
                 type='3D'):
        super().__init__()
        self.type = type
        self.info = []
        self.input_dir = input_dir
        self.num_multiphase = num_multiphase
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.out_phase_id = out_phase_id

        self.info_name = {}
        env = lmdb.open(self.input_dir, readonly=True, lock=False,
                        readahead=False, meminit=False, create=False)
        filename_list = []
        self.info = []
        with env.begin(write=False) as txn:
            for k, v in pickle.loads(txn.get(b'__keys__')).items():
                for i in v:
                    if i[0] in filename_list:
                        continue
                    else:
                        self.info.extend([i])
        env.close()
        if self.type == '3D':
            self.all_total_data_list = self.flatten_data(self.info)
        elif self.type == '2D':
            self.all_total_data_list = self.flatten_data_2D(self.info)

    def flatten_data(self, data):
        new_list = []
        for ct_name, value, (_, max_depth, width, height), affine_matrix, result_list in data:
            self.info_name[ct_name] = (value, self.num_multiphase, (0, max_depth, width, height), affine_matrix,
                                       result_list)
            new_list.append(ct_name)
        return new_list

    def open_lmdb(self):
        self.env = lmdb.open(self.input_dir, readonly=True, lock=False,
                             readahead=False, meminit=False, create=False)
        self.txn = self.env.begin(buffers=True)

    def read_slices_train(self, case_id):
        ct_min, num_multiphase, i, affine_matrix, shape = self.info_name[case_id]
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        img_all = []
        mask_all = []
        mask_vessel_all = []
        for t in range(i[0], i[1]):
            bin_data = self.txn.get(str(case_id + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            high_bytes, low_bytes, mask, mask_vessel = torch.reshape(data, (4, -1, i[-2], i[-1]))[:, :num_multiphase, :,
                                                       :]
            ct = decode_ct_data(high_bytes, low_bytes)
            ct = ct + ct_min
            img_all.append(ct)
            mask_all.append(mask)
            mask_vessel_all.append(mask_vessel)

        img = torch.stack(img_all, 1)
        mask = torch.stack(mask_all, 1)
        mask_vessel = torch.stack(mask_vessel_all, 1)
        return img, mask, mask_vessel, affine_matrix

    def read_slices_val(self, case_id):
        ct_min, num_multiphase, i, affine_matrix, shape = self.info_name[case_id]

        if not hasattr(self, 'txn'):
            self.open_lmdb()
        img_all = []
        mask_all = []
        mask_vessel_all = []
        for t in range(i[0], i[1]):
            bin_data = self.txn.get(str(case_id + "_image" + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            high_bytes, low_bytes = torch.reshape(data, (2, -1, i[-2], i[-1]))[:, :num_multiphase, :, :]
            ct = decode_ct_data(high_bytes, low_bytes)
            ct = ct + ct_min
            img_all.append(ct)
        img = torch.stack(img_all, 1)

        for t in range(0, shape[3]):
            bin_data = self.txn.get(str(case_id + "_label" + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            mask = torch.reshape(data, (-1, shape[1], shape[2]))[:num_multiphase, :, :]
            mask_all.append(mask)
        mask = torch.stack(mask_all, 1)

        for t in range(i[0], i[1]):
            bin_data = self.txn.get(str(case_id + "_label_vessel" + f"_{t}").encode())
            data = torchvision.io.decode_png(pandas.read_pickle(io.BytesIO(bin_data)))
            mask_vessel = torch.reshape(data, (-1, i[-2], i[-1]))[:num_multiphase, :, :]
            mask_vessel_all.append(mask_vessel)
        mask_vessel = torch.stack(mask_vessel_all, 1)

        return img, mask, mask_vessel, shape, affine_matrix

    def __len__(self):
        return len(self.all_total_data_list)

    def __getitem__(self, index):
        name = self.all_total_data_list[index]

        if self.dataset_type == 'train':
            image, mask, mask_vessel, affine_matrix = self.read_slices_train(name)

            image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            mask_vessel = mask_vessel.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
            sample = {"image": image, "label": mask, "label_vessel": mask_vessel}
            if self.transforms != None:
                sample = self.transforms(sample)
            return sample

        elif self.dataset_type == 'val':
            image, mask, mask_vessel, shape, affine_matrix = self.read_slices_val(name)

            if len(image) != self.num_multiphase:
                return {"image": image, "label": mask, "label_vessel": mask_vessel, "affine_matrix": affine_matrix,
                        "name": name}
            else:
                image = image.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                mask = mask.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                mask_vessel = mask_vessel.permute(0, 3, 2, 1).contiguous()[self.out_phase_id]
                sample = {"image": image, "label": mask, "label_vessel": mask_vessel, "affine_matrix": affine_matrix,
                          "name": name}
                if self.transforms != None:
                    sample = self.transforms(sample)
                return sample


class Dataset_2D(LMDBDataset):
    def __init__(self,
                 input_dir,
                 num_multiphase=1,
                 out_phase_id=[2],
                 dataset_type='train',
                 transforms=None,
                 type='2D'):
        super().__init__(input_dir, dataset_type, num_multiphase, type)

        self.type = type
        self.info = []
        self.input_dir = input_dir
        self.num_multiphase = num_multiphase
        self.dataset_type = dataset_type
        self.out_phase_id = out_phase_id
        self.transforms = transforms

    def __len__(self):
        return len(self.all_total_data_list)

    def __getitem__(self, index):
        name = self.all_total_data_list[index]
        image, mask = self.read_slice(name)
        image = image.unsqueeze(1).contiguous()[self.out_phase_id]
        mask = mask.unsqueeze(1).contiguous()[self.out_phase_id]
        if self.transforms != None:
            image, mask = self.transforms(image, mask)
        return image, mask
