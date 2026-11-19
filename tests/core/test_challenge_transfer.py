"""Transfer of a painted-mesh label to the GT point cloud (preprocess-challenges)."""

import numpy as np

from eval3d.pipeline.preprocess_challenges import _transfer_mesh_mask_to_pcd

TRIANGLE = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
FACE = np.array([[0, 1, 2]])


def test_a_point_inside_a_large_painted_triangle_is_labelled():
    # 3 mm from every vertex but on the surface: the nearest-vertex test alone drops it
    points = np.array([[3.0, 3.0, 0.0]])
    mask = np.array([True, True, True])
    assert not _transfer_mesh_mask_to_pcd(TRIANGLE, mask, points, 0.1)[0]
    assert _transfer_mesh_mask_to_pcd(TRIANGLE, mask, points, 0.1, faces=FACE)[0]


def test_the_distance_threshold_still_applies_to_the_surface():
    points = np.array([[3.0, 3.0, 0.05], [3.0, 3.0, 0.2]])
    got = _transfer_mesh_mask_to_pcd(TRIANGLE, np.array([True, True, True]), points, 0.1, faces=FACE)
    assert got.tolist() == [True, False]


def test_a_point_on_the_other_sheet_of_a_thin_wall_keeps_the_label_of_its_own_sheet():
    # painted sheet at z = 0, unpainted sheet at z = 0.3
    vertices = np.vstack([TRIANGLE, TRIANGLE + [0.0, 0.0, 0.3]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mask = np.array([True, True, True, False, False, False])
    points = np.array([[3.0, 3.0, 0.0], [3.0, 3.0, 0.3], [3.0, 3.0, 0.26]])
    assert _transfer_mesh_mask_to_pcd(vertices, mask, points, 0.1, faces=faces).tolist() == [True, False, False]


def test_inside_a_triangle_the_label_is_that_of_its_nearest_vertex():
    mask = np.array([True, False, False])
    points = np.array([[2.0, 2.0, 0.0], [7.0, 2.0, 0.0], [2.0, 7.0, 0.0]])
    assert _transfer_mesh_mask_to_pcd(TRIANGLE, mask, points, 0.1, faces=FACE).tolist() == [True, False, False]


def test_points_next_to_a_vertex_keep_the_nearest_vertex_label():
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(np.arange(6) * 0.05, np.arange(6) * 0.05), -1).reshape(-1, 2)
    vertices = np.c_[grid, np.zeros(len(grid))]
    faces = np.array([[i * 6 + j, i * 6 + j + 1, (i + 1) * 6 + j] for i in range(5) for j in range(5)]
                     + [[i * 6 + j + 1, (i + 1) * 6 + j + 1, (i + 1) * 6 + j] for i in range(5) for j in range(5)])
    mask = vertices[:, 0] < 0.12
    points = np.c_[rng.uniform(0, 0.25, (200, 2)), rng.uniform(-0.02, 0.02, 200)]
    old = _transfer_mesh_mask_to_pcd(vertices, mask, points, 0.1)
    assert (_transfer_mesh_mask_to_pcd(vertices, mask, points, 0.1, faces=faces) == old).all()
