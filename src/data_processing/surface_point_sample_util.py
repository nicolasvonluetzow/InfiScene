import numpy as np

"""
From previous project https://github.com/HaileyMai/Self-supervised-scene-generation-with-semantic-segmentation/blob/master/torch/sample_util.py
"""


def sample_point_cloud(vertices_, faces_, cat_ids, n_points_per_face, add_centers=False, uniform=False,
                       force_total_n=False, with_semantics=True):
    triangle_vertices = np.dstack([vertices_[faces_[:, 0]], vertices_[
                                  faces_[:, 1]], vertices_[faces_[:, 2]]])

    if force_total_n:
        n_points = n_points_per_face
        add_centers = False
    else:
        n_points = n_points_per_face * faces_.shape[0]

    if uniform:
        triangle_areas = 0.5 * np.linalg.norm(np.cross(triangle_vertices[:, 1, :] - triangle_vertices[:, 0, :],
                                                       triangle_vertices[:, 2, :] - triangle_vertices[:, 0, :]), axis=1)
        triangle_probabilities = triangle_areas / triangle_areas.sum()
        chosen_triangles = np.random.choice(
            range(triangle_areas.shape[0]), n_points, p=triangle_probabilities)
    else:
        chosen_triangles = np.repeat(
            np.arange(faces_.shape[0]), n_points_per_face)
    chosen_vertices = triangle_vertices[chosen_triangles, :, :]

    if with_semantics:
        category = cat_ids[chosen_triangles]

    r1 = np.random.rand(n_points, 1)
    r2 = np.random.rand(n_points, 1)
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2
    result_xyz = u * chosen_vertices[:, :, 0] + v * \
        chosen_vertices[:, :, 1] + w * chosen_vertices[:, :, 2]
    if add_centers:
        centers = (vertices_[faces_[:, 0]] +
                   vertices_[faces_[:, 1]] + vertices_[faces_[:, 2]]) / 3
        result_xyz = np.concatenate((result_xyz, centers))
        if with_semantics:
            category = np.concatenate((category, cat_ids))

    if with_semantics:
        return result_xyz, category
    else:
        return result_xyz
