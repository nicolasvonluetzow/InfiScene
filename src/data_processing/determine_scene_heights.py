import os
import trimesh
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    required=True, help='Path to 3D-Front dataset')
parser.add_argument('--output_path', type=str, default=os.path.join('..', '..', 'heights.json'),
                    help='Path to save output JSON')

args = parser.parse_args()

dataset_path = args.dataset_path

json_files = os.listdir(os.path.join(dataset_path, '3D-FRONT'))
print(len(json_files))

output_path = args.output_path

if os.path.exists(output_path):
    print('File already exists')
    exit()

# allowed_strings = ["wall", "floor", "ceil"]
heights = {}
for n_m, m in enumerate(json_files):
    print(n_m, m[:-5])

    p = os.path.join(dataset_path, '3D-FRONT', m)
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mesh_uid = []
    mesh_xyz = []
    mesh_faces = []

    for mm in data['mesh']:  # mesh refers to wall/floor/etc
        mm_type = mm['type']

        # if type contains any of the allowed strings (not case sensitive)
        # if any([s in mm_type.lower() for s in allowed_strings]):
        mesh_uid.append(mm['uid'])
        mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
        mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))

    boundary_meshes = [trimesh.Trimesh(
        vertices=xyz, faces=faces) for xyz, faces in zip(mesh_xyz, mesh_faces)]
    boundary_mesh = trimesh.util.concatenate(boundary_meshes)

    height = boundary_mesh.bounds[1][1] - boundary_mesh.bounds[0][1]
    # print(height)
    heights[m[:-5]] = height


with open(output_path, 'w') as f:
    json.dump(heights, f)
