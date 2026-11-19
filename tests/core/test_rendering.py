"""Tests for rendering module."""


import numpy as np
import pytest
import trimesh


@pytest.mark.gl
def test_offscreen_renderer_basic():
    """OffscreenRenderer should render a simple colored mesh."""
    from eval3d.core.rendering import OffscreenRenderer

    # Create simple triangle, wound to FACE the camera (normal -Z, toward an OpenCV camera at
    # the origin looking +Z): pyrender culls back faces, so [0, 1, 2] rendered nothing.
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
