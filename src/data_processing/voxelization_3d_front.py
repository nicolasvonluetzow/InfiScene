import json
import math
import os

import numpy as np
import trimesh
import argparse

from ..diffusion.utils import load_scene_or_mesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'

"""
Adapted from https://github.com/yanghtr/Sync2Gen/blob/main/dataset_3dfront/json2layout.py
"""


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

# Combine meshes into scenes defined in 3D-front json and create voxelized scenes


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    required=True, help='Path to 3D-Front dataset')
parser.add_argument('--output_path', type=str, default=os.path.join('..', 'meshes'),
                    help='Path to save output meshes')

args = parser.parse_args()
dataset_path = args.dataset_path
output_path = args.output_path

os.makedirs(output_path, exist_ok=True)

json_files = os.listdir(os.path.join(dataset_path, '3D-FRONT'))
print(len(json_files))

model_info_dict = modelInfo2dict(os.path.join(
    dataset_path, '3D-FUTURE-model', 'model_info.json'))

for n_m, m in enumerate(json_files):
    try:
        print(n_m, m[:-5])
        mesh_out_path = os.path.join(output_path, m[:-5]+'.obj')
        if os.path.exists(mesh_out_path):
            continue

        p = os.path.join(dataset_path, '3D-FRONT', m)
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

        boundary_meshes = [trimesh.Trimesh(
            vertices=xyz, faces=faces) for xyz, faces in zip(mesh_xyz, mesh_faces)]
        boundary_mesh = trimesh.util.concatenate(boundary_meshes)
        boundary_mesh.export(os.path.join(output_path, m[:-5]+'_boundary.obj'))

        room_meshes = []

        scene = data['scene']
        room = scene['room']
        for r in room:
            # if r['type'] not in room_types:
            #     continue

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
                    dataset_path, '3D-FUTURE-model', model_jid[idx])
                if not os.path.exists(p):
                    print(model_info_dict[model_jid[idx]]['category'])
                    continue

                p = os.path.join(dataset_path, '3D-FUTURE-model',
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

                meshes.append(child_mesh)

            # concatenate all meshes in a room into one mesh
            room_mesh = trimesh.util.concatenate(meshes)

            if room_mesh != []:  # workaround for some meshes
                room_meshes.append(room_mesh)

        furniture_mesh = trimesh.util.concatenate(room_meshes)
        # furniture_mesh.export(os.path.join(
        #     output_path, m[:-5]+'_furniture.obj'), file_type='obj')

        scene_mesh = trimesh.util.concatenate([boundary_mesh, furniture_mesh])
        scene_mesh.export(os.path.join(
            output_path, m[:-5]+'.obj'), file_type='obj')

        # voxelization
        # voxel_size = 0.05

        # min_extents = np.floor(
        #     np.min(scene_mesh.vertices, axis=0) / voxel_size) * voxel_size
        # max_extents = np.ceil(
        #     np.max(scene_mesh.vertices, axis=0) / voxel_size) * voxel_size

        # min_extents += voxel_size/2
        # max_extents -= voxel_size/2

        # x, y, z = np.mgrid[min_extents[0]:max_extents[0]+voxel_size:voxel_size,
        #                    min_extents[1]:max_extents[1]+voxel_size:voxel_size,
        #                    min_extents[2]:max_extents[2]+voxel_size:voxel_size]
        # sample_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        # print(sample_points.shape)

        # f = SDF(scene_mesh.vertices, scene_mesh.faces)
        # sdf_values = f(sample_points)
        # f = SDF().to('cuda')
        # faces_tens = torch.from_numpy(scene_mesh.faces).to('cuda')
        # vertices_tens = torch.from_numpy(scene_mesh.vertices).to('cuda')
        # # sdf_values = sdf(faces_tens, vertices_tens, grid_size=32).cpu().numpy()

        # sdf_grid = mesh_to_voxels(
        #     scene_mesh, 128, sign_method='depth', scan_count=400, scan_resolution=640)

        # # sdf_grid = -sdf_values.reshape(x.shape[0], x.shape[1], x.shape[2])

        # sdf_grid = np.pad(sdf_grid, ((1, 1), (1, 1), (1, 1)), constant_values=1e-4)

        # vertices, faces, normals, _ = marching_cubes(sdf_grid, level=0)

        # mesh = trimesh.Trimesh(
        #     vertices, faces, vertex_normals=normals)

        # mesh.export(file_obj=os.path.join(
        #     output_path, m[:-5]+'_sdf.obj'), file_type="obj")

        # occ_grid = scene_mesh.voxelized(voxel_size).matrix
        # occ_grid = 2 * occ_grid - 1

        # p = os.path.join(output_path, m[:-5]+'_occupancy.npy')
        # np.save(p, occ_grid)

        # vertices, faces, normals, _ = marching_cubes(occ_grid, level=0)
        # mesh = trimesh.Trimesh(
        #     vertices, faces, vertex_normals=normals)

        # mesh.export(file_obj=os.path.join(
        #     output_path, m[:-5]+'_occ.obj'), file_type="obj")
    except Exception as e:
        print(e)
        continue
