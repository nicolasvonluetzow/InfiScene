import argparse
import os
import numpy as np
import pyrender
from PIL import Image
from skimage.measure import marching_cubes
import trimesh
from tqdm import tqdm

from ..diffusion.utils import get_all_files, TRUNC_VAL, normalize_mesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def render_images(mesh, out_dir, num_views, number_offset):
    scene = pyrender.Scene(bg_color=(0.0, 255.0, 0.0, 255.0))

    mesh = pyrender.Mesh.from_trimesh(
        mesh, material=pyrender.MetallicRoughnessMaterial(), smooth=False)
    mesh_node = scene.add(mesh)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    light_pose = np.eye(4)
    light_pose[0, 0] = np.cos(2)
    light_pose[2, 2] = np.cos(2)
    light_pose[0, 2] = np.sin(2)
    light_pose[2, 0] = - np.sin(2)
    scene.add(light, light_pose)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 1.5
    scene.add(camera, pose=camera_pose)

    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    r = pyrender.OffscreenRenderer(512, 512)

    for i in range(num_views):
        # angle_x = i * 2 * np.pi / num_views
        # R_x = np.array([[1, 0, 0],
        #               [0, np.cos(angle_x), -np.sin(angle_x)],
        #               [0, np.sin(angle_x), np.cos(angle_x)]])

        angle_y = i * 2 * np.pi / num_views
        # R_z = np.array([[np.cos(angle_y), -np.sin(angle_y), 0],
        #                 [np.sin(angle_y), np.cos(angle_y), 0],
        #                 [0, 0, 1]])

        R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                        [0, 1, 0],
                        [-np.sin(angle_y), 0, np.cos(angle_y)]])

        # R = np.dot(R_z, R_x)
        R = R_y
        mesh_pose = np.eye(4)
        mesh_pose[0:3, 0:3] = R
        scene.set_pose(mesh_node, pose=mesh_pose)

        np_img, _ = r.render(scene, flags=flags)
        img = Image.fromarray(np_img)
        img.save(os.path.join(out_dir, f'image_{i + number_offset:06d}.png'))

    r.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_folder', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)

    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--mc_thresh', type=float, default=1.0)
    parser.add_argument('--num_views', type=int, default=8)

    args = parser.parse_args()

    if args.out_dir is None:
        # append _images to the df_folder name, need to care if df_folder ends with /
        args.out_dir = args.df_folder.rstrip('/') + '_images'
        print(f"Using {args.out_dir} as output folder")

    os.makedirs(args.out_dir, exist_ok=True)

    df_files = get_all_files(args.df_folder)
    df_files = [f for f in df_files if f.endswith(".npy")]
    print(f"Found {len(df_files)} files")

    if args.max_samples is not None:
        df_files = df_files[:args.max_samples]
        print(f"Using {len(df_files)} files")

    for i, df_file in tqdm(enumerate(df_files)):
        df = np.load(df_file)

        try:
            verts, faces, normals, _ = marching_cubes(df, level=args.mc_thresh)
            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals)

            normalize_mesh(mesh)

            render_images(mesh, args.out_dir, args.num_views,
                          number_offset=i*args.num_views)
        except:
            print(f"Failed to render mesh {i}, from file {df_file}")
