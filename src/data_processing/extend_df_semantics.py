import json
import math
import multiprocessing as mp
import os
import random
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse

from .surface_point_sample_util import sample_point_cloud
from ..diffusion.utils import load_scene_or_mesh

"""
Adapted from https://github.com/yanghtr/Sync2Gen/blob/main/dataset_3dfront/json2layout.py
"""

BOUNDARY_OCCUPANCY = 0.4
BOUNDARY_POINTS_PER_FACE = 5000
OBJECT_POINTS_PER_FACE = 100

super_categories = {
    'Unknown': 0,
    'Floor/Ceiling': 1,
    'Boundaries': 2,
    'Other': 3,
    'Cabinet/Shelf/Desk': 4,
    'Bed': 5,
    'Chair': 6,
    'Table': 7,
    'Sofa': 8,
    'Pier/Stool': 9,
    'Lighting': 10
}

color_table = mpl.colormaps.get_cmap('tab20')

# cube of all 27 integer coordinates in [-1, 1]^3
cube = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1],
                 [0, 1, 0], [0, 1, 1], [0, 1, -1],
                 [0, -1, 0], [0, -1, 1], [0, -1, -1],
                 [1, 0, 0], [1, 0, 1], [1, 0, -1],
                 [1, 1, 0], [1, 1, 1], [1, 1, -1],
                 [1, -1, 0], [1, -1, 1], [1, -1, -1],
                 [-1, 0, 0], [-1, 0, 1], [-1, 0, -1],
                 [-1, 1, 0], [-1, 1, 1], [-1, 1, -1],
                 [-1, -1, 0], [-1, -1, 1], [-1, -1, -1]])


def load_df_as_numpy(path):
    with open(path, 'rb') as f:
        dimX = int.from_bytes(f.read(4), byteorder='little')
        dimY = int.from_bytes(f.read(4), byteorder='little')
        dimZ = int.from_bytes(f.read(4), byteorder='little')

        voxelSize = struct.unpack('f', f.read(4))[0]

        voxelToWorld = []
        for i in range(4):
            for j in range(4):
                voxelToWorld.append(struct.unpack('f', f.read(4))[0])

        voxelToWorld = np.array(voxelToWorld).reshape(4, 4)

        data = np.fromfile(f, dtype=np.float32).reshape((dimZ, dimY, dimX))
        data = data.swapaxes(0, 2)

    return data, voxelToWorld


def read_obj_vertices(obj_path):
    ''' This is slow. Check obj file format. 
    @Returns:
        v: N_vertices x 3
    '''
    v_list = []
    with open(obj_path, 'r') as f:
        for line in f.readlines():
            if line[:2] == 'v ':
                v_list.append([float(a) for a in line.split()[1:]])
    return np.array(v_list)


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def modelInfo2dict(model_info_path):
    model_info_dict = {}
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    for v in info:
        model_info_dict[v['model_id']] = v
    return model_info_dict


def sample_points(vertices, faces, worldToVoxel, n_points_per_face=5):
    points = sample_point_cloud(vertices, faces, cat_ids=None, n_points_per_face=n_points_per_face,
                                add_centers=True, uniform=True, with_semantics=False)

    x = np.ones((points.shape[0], 4))
    x[:, :3] = points[:, :3]
    x = np.matmul(worldToVoxel, np.transpose(x))
    x = np.transpose(x)[:, :3]

    x_round = np.floor(x + np.array([0.5, 0.5, 0.5])).astype(int)

    points = (cube[None, :, :] + x_round[:, None, :]
              ).reshape(-1, 3)

    return points


def flatten_df_sem_into_floorplan(df_sem):
    "We collapse the 3D voxel grid into a 2D floorplan by taking the max value along the vertical axis, ignoring the boundaries."

    df_sem_copy = np.copy(df_sem)

    # replace boundaries with 0
    df_sem_copy[df_sem_copy == super_categories['Boundaries']] = 0
    df_sem_copy[df_sem_copy == super_categories['Floor/Ceiling']] = 0

    # take max along vertical axis
    floorplan = np.max(df_sem_copy, axis=1)

    # count boundaries in each column
    boundary_count = np.sum(df_sem == super_categories['Boundaries'], axis=1)

    # if there are majorly boundaries in a column, replace with boundary
    floorplan[boundary_count > BOUNDARY_OCCUPANCY * df_sem.shape[1]
              ] = super_categories['Boundaries']

    # if the floorplan is unknown and there is a boundary in the column, replace with floor/ceiling
    floorplan[(floorplan == super_categories['Unknown']) & (boundary_count > 0)
              ] = super_categories['Floor/Ceiling']

    return floorplan


def flatten_df_sem_into_floorplan_v2(df_sem):
    "Instead of keeping only a single value per column, we save all that apply."

    floorplan = np.zeros(
        (df_sem.shape[0], df_sem.shape[2], len(super_categories) - 1), dtype=np.bool8)

    for super_cat_name in super_categories.keys():
        if super_cat_name in ['Unknown', 'Floor/Ceiling', 'Boundaries']:
            continue

        super_cat = super_categories[super_cat_name]

        presence = np.any(df_sem == super_cat, axis=1)
        floorplan[:, :, super_cat - 1] = presence

    # boundary and floor/ceiling logic
    boundary_count = np.sum(df_sem == super_categories['Boundaries'], axis=1)

    floorplan[:, :, super_categories['Boundaries'] -
              1] = boundary_count > BOUNDARY_OCCUPANCY * df_sem.shape[1]
    floorplan[:, :, super_categories['Floor/Ceiling'] -
              1] = boundary_count > 0

    return floorplan


def extend_file(df_path, corresponding_df, json_path, m, model_info_dict, output_path_floorplan, output_path_semantic_dfs):
    if not os.path.exists(os.path.join(df_path, corresponding_df)):
        # print('--df not found')
        return

    if os.path.exists(os.path.join(output_path_floorplan, m[:-5]) + '.npy'):
        # print('--already exists')
        return

    df, voxelToWorld = load_df_as_numpy(
        os.path.join(df_path, corresponding_df))

    worldToVoxel = np.linalg.inv(voxelToWorld)

    # create empty semantic voxel grid
    df_sem = np.zeros(df.shape, dtype=np.uint8)

    p = os.path.join(json_path, '3D-FRONT', m)
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_jid = []
    model_uid = []
    model_bbox = []

    mesh_uid = []
    mesh_xyz = []
    mesh_faces = []
    # model_uid & model_jid store all furniture info of all rooms
    for ff in data['furniture']:
        if 'valid' in ff and ff['valid']:
            model_uid.append(ff['uid'])  # used to access 3D-FUTURE-model
            model_jid.append(ff['jid'])
            model_bbox.append(ff['bbox'])
    for mm in data['mesh']:  # mesh refers to wall/floor/etc
        mesh_uid.append(mm['uid'])
        mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
        mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))

    for vertices, faces in zip(mesh_xyz, mesh_faces):
        try:
            points = sample_points(
                vertices, faces, worldToVoxel, n_points_per_face=BOUNDARY_POINTS_PER_FACE)

            df_sem[points[:, 0], points[:, 1], points[:, 2]
                   ] = super_categories['Boundaries']
        except Exception as e:
            # print(e)
            continue

    scene = data['scene']
    room = scene['room']
    for r in room:

        room_id = r['instanceid']
        meshes = []
        children = r['children']
        number = 1
        for c in children:

            ref = c['ref']
            if ref not in model_uid:  # mesh (wall/floor) not furniture
                continue
            idx = model_uid.index(ref)

            p = os.path.join(
                json_path, '3D-FUTURE-model', model_jid[idx])
            if not os.path.exists(p):
                print(model_info_dict[model_jid[idx]]['category'])
                continue

            super_category = model_info_dict[model_jid[idx]]['super-category']
            super_category = super_categories[super_category]

            p = os.path.join(json_path, '3D-FUTURE-model',
                             model_jid[idx], 'raw_model.obj')
            child_mesh = load_scene_or_mesh(p)

            rot = c['rot'][1:]
            pos = c['pos']
            scale = c['scale']

            child_mesh.vertices *= scale

            dref = [0, 0, 1]
            axis = np.cross(dref, rot)
            theta = np.arccos(np.dot(dref, rot))*2

            if np.sum(axis) != 0 and not math.isnan(theta):
                R = rotation_matrix(axis, theta)
                verts = child_mesh.vertices
                verts = np.transpose(verts)
                verts = np.matmul(R, verts)
                child_mesh.vertices = np.transpose(verts)

            child_mesh.apply_translation(pos)

            points = sample_points(
                child_mesh.vertices, child_mesh.faces, worldToVoxel, n_points_per_face=OBJECT_POINTS_PER_FACE)

            df_sem[points[:, 0], points[:, 1], points[:, 2]] = super_category

    floorplan_v2 = flatten_df_sem_into_floorplan_v2(df_sem)
    # print(floorplan_v2.shape)
    # save as numpy
    output_file = os.path.join(
        output_path_floorplan, m[:-5]) + '.npy'
    print(output_file, floorplan_v2.shape)
    np.save(output_file, floorplan_v2)

    chosen_sem = np.zeros(floorplan_v2.shape[:2], dtype=np.uint8) + 15
    for super_cat in range(floorplan_v2.shape[2]):
        chosen_sem[floorplan_v2[:, :, super_cat]] = super_cat + 1

    # apply color table
    colors_floor = color_table(chosen_sem) * 255
    # save as image
    output_file = os.path.join(
        output_path_floorplan, m[:-5]) + '.png'
    print(output_file)
    Image.fromarray(colors_floor.astype(np.uint8)).save(output_file)

    # save df_sem using numpy
    output_file = os.path.join(
        output_path_semantic_dfs, m[:-5]) + '.npy'
    print(output_file, df_sem.shape)
    np.save(output_file, df_sem)


parser = argparse.ArgumentParser()
parser.add_argument('--json_path', required=True, type=str)
parser.add_argument('--df_path', required=True, type=str)
parser.add_argument('--output_path_floorplan', default=os.path.join(
    '..', '..', 'floorplans'), type=str)
parser.add_argument('--output_path_semantic_dfs', default=os.path.join(
    '..', '..', 'semantic_dfs'), type=str)

args = parser.parse_args()
json_path = args.json_path
df_path = args.df_path
output_path_floorplan = args.output_path_floorplan
output_path_semantic_dfs = args.output_path_semantic_dfs

os.makedirs(output_path_floorplan, exist_ok=True)
os.makedirs(output_path_semantic_dfs, exist_ok=True)

json_files = os.listdir(os.path.join(json_path, '3D-FRONT'))
print(len(json_files))

model_info_dict = modelInfo2dict(os.path.join(
    json_path, '3D-FUTURE-model', 'model_info.json'))


# visualize color for each super category
# set up plot with 5x5 subplots
fig, ax = plt.subplots(4, 4, figsize=(10, 10))

for super_cat_name in super_categories.keys():
    if super_cat_name in ['Unknown']:
        continue

    super_cat = super_categories[super_cat_name]
    super_cat = np.array(super_cat)

    color = color_table(super_cat)
    # plt.imshow(np.array([color] * 100).reshape(10, 10, 4))
    # plt.title(super_cat_name)
    # plt.savefig(os.path.join(
    #     output_path, super_cat_name.replace('/', '-') + '.png'))

    # select the right subplot
    ax_idx = super_cat
    ax_idx = (ax_idx // 4, ax_idx % 4)

    # plot the color
    ax[ax_idx].imshow(np.array([color] * 100).reshape(10, 10, 4))
    ax[ax_idx].set_title(super_cat_name)

plt.savefig(os.path.join(output_path_floorplan, 'color_table.png'))

# parrallelize
random.shuffle(json_files)
pool = mp.Pool(7)
for n_m, m in enumerate(json_files):
    pool.apply_async(extend_file, args=(df_path, m[:-5] + '.df', json_path,
                                        m, model_info_dict, output_path_floorplan, output_path_semantic_dfs))
pool.close()
pool.join()


# random.shuffle(json_files)
# for n_m, m in tqdm(enumerate(json_files), total=len(json_files)):
#     try:
#         extend_file(df_path, m[:-5] + '.df', json_path,
#                     m, model_info_dict, output_path)
#     except Exception as e:
#         print(e)
#         continue
#     # extend_file(df_path, m[:-5] + '.df', json_path,
#     #             m, model_info_dict, output_path)
