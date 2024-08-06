import math
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .utils import (TRUNC_VAL, load_df_as_tensor, mc_and_save)

# number of voxels added as padding during voxelization to each side of the scene
VOXELIZATION_PADDING = 5


class SceneDataset(Dataset):
    def __init__(self,
                 files,
                 augment,
                 n_chunks,
                 chunk_size,
                 num_conditions=2,
                 forced_boundary_prob=0.0,
                 surface_proximity_threshold=0.0,
                 floorplan_dir=None,
                 keep_gen_area_cond_excess=False,
                 boundary_dir=None,
                 semantic_dir=None,
                 height_json=None,
                 force_cubic=False,
                 data_downsample_factor=1,
                 is_super_scaling=False):

        super().__init__()

        self.files = files

        self.n_chunks = n_chunks
        self.chunk_size = chunk_size

        self.forced_boundary_prob = forced_boundary_prob
        self.surface_proximity_threshold = surface_proximity_threshold

        self.floorplan_dir = floorplan_dir
        self.boundary_dir = boundary_dir
        self.semantic_dir = semantic_dir
        self.height_json = json.load(
            open(height_json)) if height_json is not None else None

        self.data_downsample_factor = data_downsample_factor
        if self.data_downsample_factor > 1:
            self.chunk_size = self.chunk_size
            self.max_pool = nn.MaxPool3d(
                self.data_downsample_factor, self.data_downsample_factor)

        self.is_super_scaling = is_super_scaling

        assert not keep_gen_area_cond_excess or n_chunks == 1
        self.keep_gen_area_cond_excess = keep_gen_area_cond_excess

        assert num_conditions in [0, 2, 4]
        self.num_conditions = num_conditions

        self.augment = augment
        self.force_cubic = force_cubic

    def __len__(self):
        return len(self.files)

    def maybe_min_pool(self, data, max_pool=False):
        if self.data_downsample_factor == 1:
            return data
        else:
            return -self.max_pool(-data) if not max_pool else self.max_pool(data)

    def __getitem__(self, index):
        file = self.files[index]
        scene = load_df_as_tensor(file, truncate=True)

        floorplan = None
        if self.floorplan_dir is not None:
            try:
                filename = os.path.basename(file)
                floorplan_file = os.path.join(
                    self.floorplan_dir, filename.replace(".df", ".npy"))
                floorplan = np.load(floorplan_file)
                floorplan = torch.from_numpy(floorplan).long()
                floorplan = floorplan.permute(2, 0, 1)

                assert floorplan.shape[1] == scene.shape[0] and floorplan.shape[2] == scene.shape[2]
            except Exception as e:
                # print(
                #     f"Floorplan not read for {file}.")

                random_index = random.randint(0, len(self.files) - 1)
                return self.__getitem__(random_index)

        boundary = None
        if self.boundary_dir is not None:
            try:
                filename = os.path.basename(file)
                boundary_file = os.path.join(self.boundary_dir, filename)

                if not os.path.exists(boundary_file):
                    # try to load the boundary file with the _0 suffix
                    boundary_file = boundary_file[:-3] + "_0.df"

                boundary = load_df_as_tensor(boundary_file, truncate=True)

                assert boundary.shape == scene.shape
            except Exception as e:
                # print(
                #     f"Boundary not read for {file}.")

                random_index = random.randint(0, len(self.files) - 1)
                return self.__getitem__(random_index)

        semantic = None
        if self.semantic_dir is not None:
            try:
                filename = os.path.basename(file)
                semantic_file = os.path.join(self.semantic_dir, filename)[
                    :-3] + ".npy"

                if not os.path.exists(semantic_file):
                    # try to load the semantic file with the _0 suffix
                    semantic_file = semantic_file[:-3] + "_0.npy"

                semantic = np.load(semantic_file)
                semantic = torch.from_numpy(semantic).long()

                assert semantic.shape == scene.shape
            except Exception as e:
                # print(f"Semantic not read for {file}.")
                random_index = random.randint(0, len(self.files) - 1)
                return self.__getitem__(random_index)

        if self.augment:
            # horizontal rotation
            direction = np.random.randint(0, 4)
            scene = torch.rot90(scene, k=direction, dims=(0, 2))
            if floorplan is not None:
                # floorplan has only 2 dimensions and needs to be rotated accordingly
                floorplan = torch.rot90(floorplan, k=direction, dims=(1, 2))

            if boundary is not None:
                boundary = torch.rot90(boundary, k=direction, dims=(0, 2))

            if semantic is not None:
                semantic = torch.rot90(semantic, k=direction, dims=(0, 2))

            # horizontal mirroring
            if np.random.randint(0, 2) == 0:
                scene = torch.flip(scene, dims=(0,))
                if floorplan is not None:
                    floorplan = torch.flip(floorplan, dims=(1,))

                if boundary is not None:
                    boundary = torch.flip(boundary, dims=(0,))

                if semantic is not None:
                    semantic = torch.flip(semantic, dims=(0,))

        data = self.generate_chunks(scene, floorplan, boundary, semantic)

        if self.height_json is not None:
            # file without file type
            file = file.split("/")[-1].split(".")[0]
            height = float(self.height_json[file])

            # same length as the number of chunks
            data["heights"] = torch.tensor([height] * self.n_chunks)

        return data

    def generate_chunks(self, data, floorplan, boundary_scene, semantic):
        pad0_front, pad0_back = 0, 0
        pad1_bottom, pad1_top = 0, 0
        pad2_front, pad2_back = 0, 0

        if self.force_cubic:
            if data.shape[1] < self.chunk_size:
                pad1_top = self.chunk_size - data.shape[1]
            elif data.shape[1] > self.chunk_size:
                data = data[:, :self.chunk_size, :]
        elif self.is_super_scaling:
            # make sure chunk height is divisible by  2 ** (data_downsample_factor + 1)
            factor = 2 ** (self.data_downsample_factor + 1)
            if data.shape[1] % factor != 0:
                pad1_top = factor - (data.shape[1] % factor)

        if data.shape[0] < self.chunk_size:
            pad0_front = pad0_back = math.ceil(
                (self.chunk_size - data.shape[0]) / 2)

        if data.shape[2] < self.chunk_size:
            pad2_front = pad2_back = math.ceil(
                (self.chunk_size - data.shape[2]) / 2)

        pad_voxels = 2 * (self.chunk_size - VOXELIZATION_PADDING)

        if self.num_conditions == 0:
            dim0_min = 0
            dim0_max = data.shape[0] - self.chunk_size

            dim2_min = 0
            dim2_max = data.shape[2] - self.chunk_size
        elif self.num_conditions == 2:
            conds0, conds2 = [], []

            pad0_front += pad_voxels
            pad2_front += pad_voxels

            dim0_min = self.chunk_size
            dim0_max = data.shape[0] + pad0_front + pad0_back - self.chunk_size

            dim2_min = self.chunk_size
            dim2_max = data.shape[2] + pad2_front + pad2_back - self.chunk_size

        elif self.num_conditions == 4:
            conds0, conds2 = [], []
            condsm0, condsm2 = [], []

            pad0_front += pad_voxels
            pad2_front += pad_voxels

            pad0_back += pad_voxels
            pad2_back += pad_voxels

            dim0_min = self.chunk_size
            dim0_max = data.shape[0] + pad0_front + \
                pad0_back - 2 * self.chunk_size

            dim2_min = self.chunk_size
            dim2_max = data.shape[2] + pad2_front + \
                pad2_back - 2 * self.chunk_size

        # pad data according to the collected padding values
        data = torch.nn.functional.pad(
            data, (pad2_front, pad2_back, pad1_bottom, pad1_top, pad0_front, pad0_back), mode='constant', value=TRUNC_VAL)

        if floorplan is not None:
            floorplan = torch.nn.functional.pad(
                floorplan, (pad2_front, pad2_back, pad0_front, pad0_back), mode='constant', value=0)

        if boundary_scene is not None:
            boundary_scene = torch.nn.functional.pad(
                boundary_scene, (pad2_front, pad2_back, pad1_bottom, pad1_top, pad0_front, pad0_back), mode='constant', value=TRUNC_VAL)

        if semantic is not None:
            semantic = torch.nn.functional.pad(
                semantic, (pad2_front, pad2_back, pad1_bottom, pad1_top, pad0_front, pad0_back), mode='constant', value=0)

        chunks = []
        coarse_chunks = [] if self.is_super_scaling else None

        floorplans = [] if floorplan is not None else None
        boundary_scenes = [] if boundary_scene is not None else None
        semantic_scenes = [] if semantic is not None else None

        while len(chunks) < self.n_chunks:
            if self.keep_gen_area_cond_excess:
                pos = (dim0_min, dim2_min)
            elif random.random() < self.forced_boundary_prob:
                assert self.generate_condition
                # force the selection of a boundary chunk

                boundary_min = self.chunk_size
                boundary_max = pad_voxels

                d0_boundary = random.random() < 0.5
                pos = (random.randint(boundary_min, boundary_max), random.randint(dim2_min, dim2_max)) if d0_boundary else (
                    random.randint(dim0_min, dim0_max), random.randint(boundary_min, boundary_max))
            else:
                pos = (random.randint(dim0_min, dim0_max),
                       random.randint(dim2_min, dim2_max))

            chunk = data[pos[0]:pos[0] + self.chunk_size, :, pos[1]
                :pos[1] + self.chunk_size].clone().detach().unsqueeze(0)

            surface_proximity = torch.sum(
                chunk != TRUNC_VAL) / (chunk.shape[0] * chunk.shape[1] * chunk.shape[2])
            if surface_proximity < self.surface_proximity_threshold:
                continue

            chunk = self.maybe_min_pool(chunk)
            chunks.append(chunk)

            if self.is_super_scaling:
                coarse = -nn.MaxPool3d(2, 2)(-chunk.clone().detach())
                coarse_chunks.append(coarse)

            if boundary_scene is not None:
                if not self.keep_gen_area_cond_excess:
                    boundary_chunk = boundary_scene[pos[0]:pos[0] + self.chunk_size,
                                                    :, pos[1]:pos[1] + self.chunk_size].clone().detach().unsqueeze(0)
                else:
                    boundary_chunk = boundary_scene[pos[0]:,
                                                    :, pos[1]:].clone().detach().unsqueeze(0)
                boundary_chunk = self.maybe_min_pool(boundary_chunk)
                boundary_scenes.append(boundary_chunk)

            if floorplan is not None:
                if not self.keep_gen_area_cond_excess:
                    floorplan_chunk = floorplan[:, pos[0]:pos[0] + self.chunk_size,
                                                pos[1]:pos[1] + self.chunk_size].clone().detach()
                else:
                    floorplan_chunk = floorplan[:,
                                                pos[0]:, pos[1]:].clone().detach()
                floorplans.append(floorplan_chunk)

            if semantic is not None:
                if not self.keep_gen_area_cond_excess:
                    semantic_chunk = semantic[pos[0]:pos[0] + self.chunk_size, :,
                                              pos[1]:pos[1] + self.chunk_size].clone().detach().unsqueeze(0)
                else:
                    semantic_chunk = semantic[pos[0]:, :,
                                              pos[1]:].clone().detach().unsqueeze(0)
                semantic_chunk = self.maybe_min_pool(
                    semantic_chunk.float(), max_pool=True).long()
                semantic_scenes.append(semantic_chunk)

            if self.num_conditions in [2, 4]:
                cond0 = data[pos[0] - self.chunk_size:pos[0],
                             :, pos[1]:pos[1] + self.chunk_size].clone().detach().unsqueeze(0)
                cond0 = self.maybe_min_pool(cond0)
                conds0.append(cond0)

                cond2 = data[pos[0]:pos[0] + self.chunk_size,
                             :, pos[1] - self.chunk_size:pos[1]].clone().detach().unsqueeze(0)
                cond2 = self.maybe_min_pool(cond2)
                conds2.append(cond2)

            if self.num_conditions == 4:
                condm0 = data[pos[0] + self.chunk_size:pos[0] + 2 * self.chunk_size,
                              :, pos[1]:pos[1] + self.chunk_size].clone().detach().unsqueeze(0)
                condm0 = self.maybe_min_pool(condm0)
                condsm0.append(condm0)

                condm2 = data[pos[0]:pos[0] + self.chunk_size,
                              :, pos[1] + self.chunk_size:pos[1] + 2 * self.chunk_size].clone().detach().unsqueeze(0)
                condm2 = self.maybe_min_pool(condm2)
                condsm2.append(condm2)

        data = {"chunks": torch.stack(chunks)}

        if self.is_super_scaling:
            data["coarse_chunks"] = torch.stack(coarse_chunks)

        if self.num_conditions in [2, 4]:
            data["conds0"] = torch.stack(conds0)
            data["conds2"] = torch.stack(conds2)

        if self.num_conditions == 4:
            data["condsm0"] = torch.stack(condsm0)
            data["condsm2"] = torch.stack(condsm2)

        if floorplan is not None:
            data["floorplans"] = torch.stack(floorplans)

        if boundary_scene is not None:
            data["boundaries"] = torch.stack(boundary_scenes)

        if semantic is not None:
            data["semantic"] = torch.stack(semantic_scenes)

        return data


def chunk_collate_fn(batch):
    # custom collate function to generate chunks
    max_height = max([b["chunks"].shape[-2] for b in batch])
    # pad all scenes to the same height
    chunks, heights = [], []
    conds0, conds2 = [], []
    condsm0, condsm2 = [], []

    coarse_chunks = []
    boundaries = []
    semantics = []

    for b in batch:
        # height of the chunk before any padding - even voxelization padding
        if "heights" in b:
            heights.append(b["heights"])

        # technical height of the chunk
        curr_height = b["chunks"].shape[-2]
        # we pad in the height dimension to make all chunks the same height, always pad from the top
        padding_tuple = (0, 0, 0, max_height - curr_height)
        coarse_padding_tuple = (0, 0, 0, max_height // 2 - curr_height // 2)

        chunks.append(torch.nn.functional.pad(
            b["chunks"], padding_tuple, mode='constant', value=TRUNC_VAL))

        if "coarse_chunks" in b:
            coarse_chunks.append(torch.nn.functional.pad(
                b["coarse_chunks"], coarse_padding_tuple, mode='constant', value=TRUNC_VAL))

        if "boundaries" in b:
            boundaries.append(torch.nn.functional.pad(
                b["boundaries"], padding_tuple, mode='constant', value=TRUNC_VAL))

        if "semantic" in b:
            semantics.append(torch.nn.functional.pad(
                b["semantic"], padding_tuple, mode='constant', value=TRUNC_VAL))

        if "conds0" in b:
            conds0.append(torch.nn.functional.pad(
                b["conds0"], padding_tuple, mode='constant', value=TRUNC_VAL))
        if "conds2" in b:
            conds2.append(torch.nn.functional.pad(
                b["conds2"], padding_tuple, mode='constant', value=TRUNC_VAL))
        if "condsm0" in b:
            condsm0.append(torch.nn.functional.pad(
                b["condsm0"], padding_tuple, mode='constant', value=TRUNC_VAL))
        if "condsm2" in b:
            condsm2.append(torch.nn.functional.pad(
                b["condsm2"], padding_tuple, mode='constant', value=TRUNC_VAL))

    data = {"chunks": torch.cat(chunks, dim=0).float()}

    if len(heights) > 0:
        data["heights"] = torch.cat(heights, dim=0).float()

    if len(conds0) > 0:
        data["conds0"] = torch.cat(conds0, dim=0).float()
    if len(conds2) > 0:
        data["conds2"] = torch.cat(conds2, dim=0).float()
    if len(condsm0) > 0:
        data["condsm0"] = torch.cat(condsm0, dim=0).float()
    if len(condsm2) > 0:
        data["condsm2"] = torch.cat(condsm2, dim=0).float()

    if "floorplans" in batch[0]:
        data["floorplans"] = torch.cat(
            [b["floorplans"] for b in batch], dim=0).long()

    if len(coarse_chunks) > 0:
        data["coarse_chunks"] = torch.cat(
            coarse_chunks, dim=0).float()

    if len(boundaries) > 0:
        data["boundaries"] = torch.cat(boundaries, dim=0).float()

    if len(semantics) > 0:
        data["semantic"] = torch.cat(semantics, dim=0).long()

    return data
