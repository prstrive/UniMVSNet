import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.data_io import *


def motion_blur(img: np.ndarray, max_kernel_size=3):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
    center = int((ksize - 1) / 2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return img


class BlendedMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=128, interval_scale=1.06):
        super(BlendedMVSDataset, self).__init__()

        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.metas = self.build_list()
        self.transform = transforms.ColorJitter(brightness=0.25, contrast=(0.3, 1.5))

    def build_list(self):
        metas = []
        proj_list = open(self.listfile).read().splitlines()

        for data_name in proj_list:
            dataset_folder = os.path.join(self.datapath, data_name)

            # read cluster
            cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
            cluster_lines = open(cluster_path).read().splitlines()
            image_num = int(cluster_lines[0])

            # get per-image info
            for idx in range(0, image_num):

                ref_id = int(cluster_lines[2 * idx + 1])
                cluster_info = cluster_lines[2 * idx + 2].rstrip().split()
                total_view_num = int(cluster_info[0])
                if total_view_num < self.nviews - 1:
                    continue

                src_ids = [int(x) for x in cluster_info[1::2]]

                metas.append((data_name, ref_id, src_ids))

        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode == "train":
            img = self.transform(img)
            img = motion_blur(np.array(img, dtype=np.float32))

        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.0
        return np_img

    def read_cam(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        # depth_sample_num = float(lines[11].split()[2])
        # depth_max = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_mask(self, filename):
        masked_img = np.array(Image.open(filename), dtype=np.float32)
        mask = np.any(masked_img > 10, axis=2).astype(np.float32)

        h, w = mask.shape
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask,
        }
        return mask_ms

    def read_depth_and_mask(self, filename, depth_min):
        # read pfm depth file
        # (576, 768)
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        mask = np.array(depth >= depth_min, dtype=np.float32)

        h, w = depth.shape
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask,
        }
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth,
        }
        return depth_ms, mask_ms

    def __getitem__(self, idx):
        data_name, ref_id, src_ids = self.metas[idx]
        view_ids = [ref_id] + src_ids[:self.nviews - 1]

        imgs = []
        img_paths = []
        proj_matrices = []
        mask_ms, depth_ms, depth_values = None, None, None

        for i, vid in enumerate(view_ids):
            img_path = os.path.join(self.datapath, data_name, 'blended_images', '%08d.jpg' % vid)
            cam_path = os.path.join(self.datapath, data_name, 'cams', '%08d_cam.txt' % vid)
            img_paths.append(img_path)

            img = self.read_img(img_path)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam(cam_path)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            imgs.append(img)
            proj_matrices.append(proj_mat)

            if i == 0:
                ref_depth_path = os.path.join(self.datapath, data_name, 'rendered_depth_maps', '%08d.pfm' % vid)
                depth_ms, mask_ms = self.read_depth_and_mask(ref_depth_path, depth_min)

                # ref_masked_img_path = os.path.join(self.datapath, data_name, 'blended_images', '%08d_masked.jpg' % vid)
                # mask_ms = self.read_mask(ref_masked_img_path)

                # -0.5 to prevent blendedmvs bug
                # get depth values
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        return {"imgs": imgs,
                "img_paths": img_paths,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask_ms}
