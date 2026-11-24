"""Rendered normal maps, angular error and MAE (eval3d.core.normals)."""

import numpy as np
import pytest
import trimesh

from eval3d.core.normals import (
    ANGULAR_ERROR_MAX_DEG,
    EXCLUDED_RGB,
    SINGLE_RGB,
    angular_error_deg,
    color_angular_error,
    encode_normals_rgb,
    normal_error_stats,
    pixel_categories,
    render_normal_map,
    scale_intrinsics,
)


def _K(f=10.0, cx=5.0, cy=5.0):
    K = np.eye(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = cx, cy
    return K


def _plane(z=10.0, a=100.0, xmin=None):
    """Square plane z = const facing the camera at the origin (normal -z, i.e. towards it)."""
    x0 = -a if xmin is None else xmin
    v = np.array([[x0, -a, z], [x0, a, z], [a, a, z], [a, -a, z]], dtype=float)
    return trimesh.Trimesh(v, [[0, 1, 2], [0, 2, 3]], process=False)


def test_plane_facing_camera_is_opengl_blue():
    r = render_normal_map(_plane(), _K(), np.eye(4), (10, 10))
    assert r.mask.all()
    np.testing.assert_allclose(r.normals[r.mask], [[0.0, 0.0, 1.0]] * 100, atol=1e-6)
    # Hit points lie on the plane
    np.testing.assert_allclose(r.points[r.mask][:, 2], 10.0, atol=1e-6)


@pytest.mark.parametrize("pixel_centre, cx, first_hit_col", [
    # Plane covers world x >= 0. Pixel col i has its centre at x = i + pixel_centre.
    (0.0, 4.75, 5),   # col 4 at -0.75 px (miss), col 5 at +0.25 px (hit)
    (0.0, 4.25, 5),   # col 4 at -0.25 px (miss)
    (0.5, 4.25, 4),   # col 4 centre at 4.5 -> +0.25 px (hit)
])
def test_pixel_centre_convention(pixel_centre, cx, first_hit_col):
    r = render_normal_map(_plane(xmin=0.0), _K(cx=cx), np.eye(4), (10, 10), pixel_centre=pixel_centre)
    cols = np.where(r.mask.any(axis=0))[0]
    assert cols[0] == first_hit_col


def test_tilted_plane_normal_in_camera_frame():
    """Camera rotated by 30 deg about its x axis: the frontal plane's normal turns accordingly."""
    th = np.radians(30.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Rt = np.eye(4)
    Rt[:3, :3] = Rx  # world -> camera, camera centre at the world origin
    # Plane normal in the world: -z ; seen in the camera (OpenCV): Rx @ (0,0,-1)
    n_cv = Rx @ np.array([0.0, 0.0, -1.0])
    expected_gl = n_cv * np.array([1.0, -1.0, -1.0])
    r = render_normal_map(_plane(a=1000.0), _K(), Rt, (10, 10))
    assert r.mask.all()
    np.testing.assert_allclose(r.normals[r.mask], np.tile(expected_gl, (100, 1)), atol=1e-6)


def test_smooth_normals_follow_the_sphere():
    """Interpolated vertex normals: much closer to the analytic sphere normal than the facets."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation([0.0, 0.0, 5.0])
    K = _K(f=200.0, cx=50.0, cy=50.0)
    r = render_normal_map(sphere, K, np.eye(4), (100, 100))
    hit = r.mask
    assert hit.sum() > 3000
    analytic_cv = r.points[hit] - np.array([0.0, 0.0, 5.0])
    analytic_cv /= np.linalg.norm(analytic_cv, axis=1, keepdims=True)
    analytic_gl = analytic_cv * np.array([1.0, -1.0, -1.0])
    err = angular_error_deg(r.normals[hit], analytic_gl)
    assert err.mean() < 0.5
    # Facet normals of this sphere are off by ~1-2 deg on average: the test discriminates.
    facets = sphere.face_normals[r.face_index[hit]] * np.array([1.0, -1.0, -1.0])
    assert angular_error_deg(facets, analytic_gl).mean() > 2 * err.mean()


def test_bbox_is_a_crop_of_the_full_render():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere.apply_translation([0.2, -0.1, 5.0])
    K = _K(f=200.0, cx=50.0, cy=50.0)
    full = render_normal_map(sphere, K, np.eye(4), (100, 100))
    bbox = (20, 70, 30, 90)  # rmin, rmax, cmin, cmax (inclusive), as crop_to_bbox
    crop = render_normal_map(sphere, K, np.eye(4), (100, 100), bbox=bbox)
    assert crop.mask.shape == (51, 61)
    np.testing.assert_array_equal(crop.mask, full.mask[20:71, 30:91])
    np.testing.assert_allclose(crop.normals, full.normals[20:71, 30:91], atol=1e-6)


def test_nearest_vertex_maps_pixels_to_vertex_flags():
    r = render_normal_map(_plane(), _K(), np.eye(4), (10, 10))
    # Left half of the image sees the left vertices (x = -100)
    left = r.nearest_vertex[:, 0]
    assert set(np.unique(left)) <= {0, 1}
    assert (r.nearest_vertex[~r.mask] == -1).all()


def test_scale_intrinsics_keeps_the_pixel_grid_consistent():
    K = _K(f=100.0, cx=50.0, cy=40.0)
    # Centre-at-0 convention: cx_s = s * cx - (1 - s) / 2
    Ks = scale_intrinsics(K, 0.25, pixel_centre=0.0)
    assert Ks[0, 0] == pytest.approx(25.0)
    assert Ks[0, 2] == pytest.approx(0.25 * 50.0 - 0.375)
    # Centre-at-0.5 convention (pixel edges at integers): plain scaling
    Ks = scale_intrinsics(K, 0.25, pixel_centre=0.5)
    assert Ks[0, 2] == pytest.approx(12.5)
    assert Ks[1, 2] == pytest.approx(10.0)


def test_angular_error_deg():
    a = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=float)
    b = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    np.testing.assert_allclose(angular_error_deg(a, b), [0.0, 90.0, 180.0], atol=1e-6)


def test_pixel_categories_and_stats():
    gt_mask = np.array([[1, 1, 1, 1, 0, 0]], dtype=bool)
    rec_mask = np.array([[1, 1, 1, 0, 1, 0]], dtype=bool)
    excluded = np.array([[0, 0, 1, 0, 0, 0]], dtype=bool)
    cat = pixel_categories(gt_mask, rec_mask, excluded)
    # both / both / excluded / gt only / rec only / background
    assert cat["valid"].tolist() == [[True, True, False, False, False, False]]
    assert cat["excluded"].tolist() == [[False, False, True, False, False, False]]
    assert cat["single"].tolist() == [[False, False, False, True, True, False]]
    err = np.array([[10.0, 30.0, 80.0, 0.0, 0.0, 0.0]])
    s = normal_error_stats(err, cat, gt_mask, excluded)
    assert s["mae_deg"] == pytest.approx(20.0)
    assert s["median_deg"] == pytest.approx(20.0)
    assert s["n_valid"] == 2
    # Non-excluded GT pixels: 3 (cols 0, 1, 3), of which 2 are valid
    assert s["gt_coverage"] == pytest.approx(2 / 3)


def test_stats_empty_view_is_nan_not_an_error():
    z = np.zeros((2, 2), dtype=bool)
    cat = pixel_categories(z, z, z)
    s = normal_error_stats(np.zeros((2, 2)), cat, z, z)
    assert s["n_valid"] == 0
    assert np.isnan(s["mae_deg"])


def test_encode_normals_rgb():
    n = np.array([[[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]]])
    n[0, 1] /= np.sqrt(2)
    mask = np.array([[True, False]])
    rgb = encode_normals_rgb(n, mask, background=(255, 255, 255))
    assert rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [128, 128, 255]
    assert rgb[0, 1].tolist() == [255, 255, 255]


def test_color_angular_error():
    err = np.array([[0.0, ANGULAR_ERROR_MAX_DEG, 90.0, 5.0, 5.0, 5.0]])
    cat = {
        "valid": np.array([[1, 1, 1, 0, 0, 0]], dtype=bool),
        "excluded": np.array([[0, 0, 0, 1, 0, 0]], dtype=bool),
        "single": np.array([[0, 0, 0, 0, 1, 0]], dtype=bool),
    }
    rgb, cb = color_angular_error(err, cat, cmap="viridis", background=(255, 255, 255))
    import matplotlib
    vir = matplotlib.colormaps["viridis"]
    np.testing.assert_allclose(rgb[0, 0], np.round(np.array(vir(0.0)[:3]) * 255), atol=1)
    np.testing.assert_allclose(rgb[0, 1], np.round(np.array(vir(1.0)[:3]) * 255), atol=1)
    np.testing.assert_array_equal(rgb[0, 2], rgb[0, 1])  # saturated above the bound
    assert rgb[0, 3].tolist() == list(EXCLUDED_RGB)
    assert rgb[0, 4].tolist() == list(SINGLE_RGB)
    assert rgb[0, 5].tolist() == [255, 255, 255]
    assert cb == {"vmin": 0.0, "vmax": ANGULAR_ERROR_MAX_DEG, "cmap": "viridis", "label": "Angular error (°)"}


def test_angular_error_is_exact_at_small_angles():
    a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    th = np.radians(1e-3)
    b = np.array([np.sin(th), 0.0, np.cos(th)], dtype=np.float32)
    assert angular_error_deg(a, b) == pytest.approx(1e-3, rel=1e-2)
    assert angular_error_deg(a, a) == 0.0


# A face seen from behind (z-buffer surface whose winding turns away from the camera) is compared with its normal
# flipped, for the GT and the reconstruction alike (Baptiste, 24/11).

def _away(mesh):
    out = mesh.copy()
    out.faces = out.faces[:, ::-1]
    return out


def test_back_face_normal_is_flipped_towards_the_camera():
    r = render_normal_map(_away(_plane()), _K(), np.eye(4), (10, 10))
    assert r.mask.all() and r.back_face.all()
    np.testing.assert_allclose(r.normals[r.mask], [[0.0, 0.0, 1.0]] * 100, atol=1e-6)


def test_front_face_is_not_a_back_face():
    r = render_normal_map(_plane(), _K(), np.eye(4), (10, 10))
    assert not r.back_face.any()


def test_flip_can_be_turned_off():
    r = render_normal_map(_away(_plane()), _K(), np.eye(4), (10, 10), flip_back_faces=False)
    np.testing.assert_allclose(r.normals[r.mask], [[0.0, 0.0, -1.0]] * 100, atol=1e-6)


def test_inside_out_surface_has_the_same_normal_map():
    """Open cap wound either way: same visible surface, same compared normals (zero angular error)."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation([0.0, 0.0, 5.0])
    sphere.update_faces(sphere.triangles_center[:, 2] < 5.0)
    sphere.remove_unreferenced_vertices()
    K = _K(f=200.0, cx=50.0, cy=50.0)
    a = render_normal_map(sphere, K, np.eye(4), (100, 100))
    b = render_normal_map(_away(sphere), K, np.eye(4), (100, 100))
    np.testing.assert_array_equal(a.mask, b.mask)
    assert b.back_face[b.mask].all() and not a.back_face.any()
    assert angular_error_deg(a.normals[a.mask], b.normals[b.mask]).max() < 1e-3
