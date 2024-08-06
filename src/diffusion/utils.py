import os
import random
import struct

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes
from PIL import Image
import matplotlib as mpl

TRUNC_VAL = 3.0


def save_chunk_with_conds_as_np(path, chunk, cond0, cond2):
    chunk = chunk.squeeze(0).cpu().numpy()
    cond0 = cond0.squeeze(0).cpu().numpy()
    cond2 = cond2.squeeze(0).cpu().numpy()

    # combine into one array along new axis
    cond_chunk = np.stack((chunk, cond0,  cond2), axis=0)

    with open(path, 'wb') as f:
        np.save(f, cond_chunk)


def load_chunk_with_conds_as_tensor(path, truncate=True):
    # load chunk saved by previous method
    with open(path, 'rb') as f:
        cond_chunk = np.load(f)

    chunk = cond_chunk[0]
    cond0 = cond_chunk[1]
    cond2 = cond_chunk[2]

    chunk = torch.from_numpy(chunk)
    cond0 = torch.from_numpy(cond0)
    cond2 = torch.from_numpy(cond2)

    if truncate:
        chunk = truncate_tensor(chunk)
        cond0 = truncate_tensor(cond0)
        cond2 = truncate_tensor(cond2)

    return chunk, cond0, cond2


def load_df_as_tensor(path, truncate=True):
    with open(path, 'rb') as f:
        dimX = int.from_bytes(f.read(4), byteorder='little')
        dimY = int.from_bytes(f.read(4), byteorder='little')
        dimZ = int.from_bytes(f.read(4), byteorder='little')

        voxelSize = struct.unpack('f', f.read(4))[0]

        voxelToWorld = []
        for i in range(4):
            for j in range(4):
                voxelToWorld.append(struct.unpack('f', f.read(4))[0])

        try:
            data = np.fromfile(f, dtype=np.float32).reshape((dimZ, dimY, dimX))
        except:
            print(f"Error loading file {path}")
            raise
        data = data.swapaxes(0, 2)

    data = torch.from_numpy(data)
    if truncate:
        data = truncate_tensor(data)

    # Throw error if there are NaNs
    if torch.isnan(data).any():
        raise ValueError(f"Data contains NaNs: {path}")

    return data


def set_random_seeds(seed=117):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint_or_state_dict(filename, device, model):
    if filename.endswith(".sd"):
        model.load_state_dict(torch.load(filename))
        model.to(device)
    else:
        load_checkpoint(filename, device, model)


def save_checkpoint(filename, model=None, optimizer=None, lr_scheduler=None):
    print("Saving checkpoint.")
    checkpoint = {}
    if model is not None:
        checkpoint["model"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

    torch.save(checkpoint, filename)


def load_checkpoint(filename, device, model=None, optimizer=None, lr_scheduler=None):
    print("Loading checkpoint.")
    checkpoint = torch.load(filename, map_location=device)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


def truncate_tensor(sample):
    # for DF we should only have positive values
    return torch.clamp(sample, 0, TRUNC_VAL)


def normalize_mesh(mesh: trimesh.Trimesh):
    """Mesh will be centered on the origin and scaled s.t. the largest dimension tightly fits into [-1,1].

    Args:
        mesh (trimesh.Trimesh): The mesh to normalize. Changed in-place.
    """
    bbox_min, bbox_max = mesh.bounds
    center = (bbox_min + bbox_max) / 2
    max_dim = max(bbox_max - bbox_min)

    mesh.vertices -= center
    mesh.vertices /= max_dim


def mc_and_save(sample, p, threshold=1.):
    try:
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().detach().squeeze().numpy()

        verts, faces, normals, _ = marching_cubes(sample, threshold)

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            process=False)

        mesh.export(p)
    except Exception as e:
        print(f"Mesh {p} invalid, {e}")
        return


def get_all_files(root: str):
    root_list = os.listdir(root)
    all = []

    for entry in root_list:
        f = os.path.join(root, entry)
        if os.path.isdir(f):
            all += get_all_files(f)
        else:
            all.append(f)

    return sorted(all)


def get_random_bb(x1, x2, x3):
    def get_start_stop(x):
        p1 = random.randint(0, x // 2)
        p2 = random.randint(p1 + x // 2, x)

        return p1, p2

    x11, x12 = get_start_stop(x1)
    x21, x22 = get_start_stop(x2)
    x31, x32 = get_start_stop(x3)

    return x11, x12, x21, x22, x31, x32


def load_scene_or_mesh(path: str) -> trimesh.Trimesh:
    """Loads a path as a trimesh mesh. If it is a scene, convert the scene to a mesh.

    Args:
        path (str): The path of the scene or mesh.

    Returns:
        trimesh.Trimesh: The loaded mesh.
    """

    sorm = trimesh.load(path, process=False)
    if isinstance(sorm, trimesh.Scene):
        geometries = [trimesh.Trimesh(
            vertices=geometry.vertices, faces=geometry.faces) for geometry in sorm.geometry.values()]
        return trimesh.util.concatenate(geometries)
    else:  # is mesh
        return sorm


def save_floorplan(floorplan, save_path):
    """Saves a floorplan as an image. Uses the tab20 colormap.

    Args:
        floorplan (np.ndarray): The floorplan to save, with shape (super_categories, height, width). Must be a numpy array.
        save_path (str): The path to save the floorplan to.
    """

    if not isinstance(floorplan, np.ndarray):
        #assume tensor
        floorplan = floorplan.cpu().numpy().squeeze()

    color_table = mpl.colormaps.get_cmap('tab20')

    chosen_sem = np.zeros(
        (floorplan.shape[1], floorplan.shape[2]), dtype=np.uint8) + 15
    for super_cat in range(floorplan.shape[0]):
        chosen_sem[floorplan[super_cat, :, :] == 1
                   ] = super_cat + 1

    colors_floor = color_table(chosen_sem) * 255
    Image.fromarray(colors_floor.astype(
        np.uint8)).save(save_path)
