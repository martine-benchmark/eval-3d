"""Tests for rendering module."""


import numpy as np
import pytest
import trimesh


@pytest.mark.gl
def test_offscreen_renderer_basic():
    """OffscreenRenderer should render a simple colored mesh."""
    from eval3d.core.rendering import OffscreenRenderer

    # Create simple triangle, wound to FACE the camera (normal -Z, toward an OpenCV camera at
    # the origin looking +Z).
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8
    )

    # Simple intrinsics
    K = np.array(
        [
            [500, 0, 320, 0],
            [0, 500, 240, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Identity pose (looking at +Z)
    Rt = np.eye(4, dtype=np.float32)

    with OffscreenRenderer(640, 480, K) as renderer:
        image = renderer.render_mesh(mesh, Rt)

        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        # Should have some non-white pixels (the triangle)
        assert not np.all(image == 255)


def test_add_colorbar():
    """add_colorbar should add a colorbar to an image."""
    from eval3d.core.rendering import add_colorbar

    # Create simple 100x100 white image
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    result = add_colorbar(image, vmin=0.0, vmax=10.0, cmap="jet", label="Distance [mm]")

    # Result should be wider (colorbar on right)
    assert result.shape[0] == 100  # Same height
    assert result.shape[1] > 100   # Wider
    assert result.shape[2] == 3


def test_crop_to_content():
    """crop_to_content should crop to bounding box of content."""
    from eval3d.core.rendering import crop_to_content

    # Create 100x100 white image with small red square in center
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[40:60, 40:60] = [255, 0, 0]  # Red square

    cropped = crop_to_content(image, margin=5)

    # Should be approximately 30x30 (20x20 content + 5 margin each side)
    assert cropped.shape[0] == 30
    assert cropped.shape[1] == 30


@pytest.mark.gl
def test_render_views(tmp_path):
    """render_views should render multiple camera views to files."""
    from eval3d.core.rendering import render_views

    # Create simple mesh
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = np.full((3, 4), [200, 200, 200, 255], dtype=np.uint8)

    # Mock cameras dict
    K = np.array([
        [500, 0, 320, 0],
        [0, 500, 240, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    Rt = np.eye(4, dtype=np.float32)

    cameras = {
        "K": np.array([K, K]),
        "Rt": np.array([Rt, Rt]),
    }

    output_dir = tmp_path / "renders"
    # render_views returns (paths, crop_bboxes) since the crop-bbox sharing.
    paths, _ = render_views(mesh, cameras, view_indices=[0], output_dir=output_dir, scale=1.0)

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"


def _tilted_square(color):
    """A square facing an OpenCV camera at the origin, tilted 60 degrees from the image plane, of one color."""
    c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
    vertices = np.array([[-1, -1, 4 + s], [1, -1, 4 - s], [1, 1, 4 - s], [-1, 1, 4 + s]], dtype=np.float32)
    vertices[:, 0] *= c
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 2, 1], [0, 3, 2]]), process=False)
    mesh.visual.vertex_colors = np.tile(np.array([*color, 255], dtype=np.uint8), (4, 1))
    return mesh


_K = np.array([[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


@pytest.mark.gl
def test_colored_render_keeps_the_exact_vertex_colors():
    """A heatmap is read against its colorbar: a colored render must not be shaded by the lighting."""
    from eval3d.core.rendering import OffscreenRenderer

    color = (68, 1, 84)   # viridis at 0
    with OffscreenRenderer(640, 480, _K) as renderer:
        image = renderer.render_mesh(_tilted_square(color), np.eye(4, dtype=np.float32), lighting_mode="colored")
    drawn = image[np.any(image != 255, axis=-1)]
    assert len(drawn) > 1000
    assert np.abs(drawn.astype(int) - color).max() <= 1


@pytest.mark.gl
def test_uniform_render_is_still_shaded():
    """The grey mesh render shows the surface relief: its lighting stays."""
    from eval3d.core.rendering import OffscreenRenderer

    with OffscreenRenderer(640, 480, _K) as renderer:
        image = renderer.render_mesh(_tilted_square((200, 200, 200)), np.eye(4, dtype=np.float32), lighting_mode="uniform")
    drawn = image[np.any(image != 255, axis=-1)]
    assert len(drawn) > 1000 and np.abs(drawn.astype(int) - 200).max() > 5


# Back faces: the renders show the z-buffer surface (what the normal maps ray cast), a face seen from behind included;
# only its shading uses the flipped normal.

def _flipped(mesh):
    """Same surface, opposite winding: every face now turns its back to the camera."""
    out = mesh.copy()
    out.faces = out.faces[:, ::-1]
    return out


@pytest.mark.gl
def test_colored_render_draws_back_faces_with_their_colors():
    from eval3d.core.rendering import OffscreenRenderer

    color = (68, 1, 84)
    with OffscreenRenderer(640, 480, _K) as renderer:
        image = renderer.render_mesh(_flipped(_tilted_square(color)), np.eye(4, dtype=np.float32), lighting_mode="colored")
    drawn = image[np.any(image != 255, axis=-1)]
    assert len(drawn) > 1000
    assert np.abs(drawn.astype(int) - color).max() <= 1


@pytest.mark.gl
def test_uniform_render_shades_a_back_face_as_its_front():
    """A face seen from behind is lit with its normal flipped: same image as the same face wound towards the camera."""
    from eval3d.core.rendering import OffscreenRenderer

    sq = _tilted_square((200, 200, 200))
    with OffscreenRenderer(640, 480, _K) as renderer:
        front = renderer.render_mesh(sq, np.eye(4, dtype=np.float32), lighting_mode="uniform")
        back = renderer.render_mesh(_flipped(sq), np.eye(4, dtype=np.float32), lighting_mode="uniform")
    assert np.any(front != 255)
    assert np.abs(front.astype(int) - back.astype(int)).max() <= 1


def _plane(z, half, color, towards_camera):
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], dtype=np.float32)
    f = np.array([[0, 2, 1], [0, 3, 2]]) if towards_camera else np.array([[0, 1, 2], [0, 2, 3]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual.vertex_colors = np.tile(np.array([*color, 255], dtype=np.uint8), (4, 1))
    return m


@pytest.mark.gl
@pytest.mark.parametrize("lighting_mode", ["colored", "uniform"])
def test_a_nearer_back_face_hides_what_is_behind_it(lighting_mode):
    """z-buffer, not culling: a back face in front of a front face is the visible surface."""
    from eval3d.core.rendering import OffscreenRenderer

    near_back = _plane(3.0, 0.5, (255, 0, 0), towards_camera=False)
    far_front = _plane(6.0, 3.0, (0, 0, 255), towards_camera=True)
    scene = trimesh.util.concatenate([near_back, far_front])
    with OffscreenRenderer(640, 480, _K) as renderer:
        image = renderer.render_mesh(scene, np.eye(4, dtype=np.float32), lighting_mode=lighting_mode)
    centre = image[240, 320].astype(int)
    assert centre[0] > centre[2], f"centre pixel {centre}: the far plane shows through the near back face"


@pytest.mark.gl
def test_render_support_matches_the_ray_cast_normal_map():
    """Open inside-out cap (near half of a sphere, faces wound away): the render covers the pixels the normal map ray
    casts. A closed inside-out sphere would not test anything, its far half facing the camera."""
    from eval3d.core.normals import render_normal_map
    from eval3d.core.rendering import OffscreenRenderer

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.apply_translation([0.1, -0.2, 5.0])
    sphere = _flipped(sphere)
    sphere.update_faces(sphere.triangles_center[:, 2] < 5.0)
    sphere.remove_unreferenced_vertices()
    sphere.visual.vertex_colors = np.tile(np.array([68, 1, 84, 255], dtype=np.uint8), (len(sphere.vertices), 1))
    K = np.array([[200, 0, 64, 0], [0, 200, 48, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    with OffscreenRenderer(128, 96, K) as renderer:
        image = renderer.render_mesh(sphere, np.eye(4, dtype=np.float32), lighting_mode="colored")
    drawn = np.any(image != 255, axis=-1)
    # pyrender samples pixel (r, c) at (c + 0.5, r + 0.5): same rays as pixel_centre=0.5
    rays = render_normal_map(sphere, K, np.eye(4), (96, 128), pixel_centre=0.5).mask
    assert rays.sum() > 1000
    iou = (drawn & rays).sum() / (drawn | rays).sum()
    assert iou > 0.99


# Pixel convention: pyrender samples pixel (r, c) at (c + 0.5, r + 0.5); the cameras give the image position of each
# pixel centre (c + pixel_centre, 0 for AliceVision). The renders must use the same pixel grid as the ray-cast maps.

def test_pyrender_intrinsics_shift_the_principal_point():
    from eval3d.core.rendering import pyrender_intrinsics

    K = np.array([[10, 0, 50, 0], [0, 10, 40, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    Kp = pyrender_intrinsics(K, pixel_centre=0.0)
    assert (Kp[0, 2], Kp[1, 2]) == (50.5, 40.5)
    assert (Kp[0, 0], Kp[1, 1]) == (10, 10)
    np.testing.assert_array_equal(pyrender_intrinsics(K, pixel_centre=0.5), K)


@pytest.mark.gl
@pytest.mark.parametrize("pixel_centre", [0.0, 0.5])
def test_render_and_rays_cover_the_same_pixels_of_a_thin_bar(pixel_centre):
    """Bar 2 px wide with edges between pixel centres: half a pixel off would shift it by a whole column."""
    from eval3d.core.normals import render_normal_map
    from eval3d.core.rendering import OffscreenRenderer, pyrender_intrinsics

    x0, x1 = 0.25 + pixel_centre, 2.25 + pixel_centre  # image x = X + cx with f = z = 10
    v = np.array([[x0, -20, 10], [x1, -20, 10], [x1, 20, 10], [x0, 20, 10]], dtype=np.float64)
    bar = trimesh.Trimesh(vertices=v, faces=[[0, 2, 1], [0, 3, 2]], process=False)
    bar.visual.vertex_colors = np.tile(np.array([68, 1, 84, 255], dtype=np.uint8), (4, 1))
    K = np.array([[10, 0, 5, 0], [0, 10, 5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    rays = render_normal_map(bar, K, np.eye(4), (10, 12), pixel_centre=pixel_centre).mask
    with OffscreenRenderer(12, 10, pyrender_intrinsics(K, pixel_centre)) as renderer:
        image = renderer.render_mesh(bar, np.eye(4, dtype=np.float32), lighting_mode="colored")
    drawn = np.any(image != 255, axis=-1)
    assert np.where(rays.any(axis=0))[0].tolist() == [6, 7]
    np.testing.assert_array_equal(drawn, rays)
