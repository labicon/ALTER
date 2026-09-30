import numpy as np
import pytest
import torch

from envs.arm.sample.temporal_observation import TemporalObservationBuffer
from envs.arm.train.temporal_images import prepare_temporal_views
from src.image_dit import ImageDiT
from src.temporal import frame_indices, frame_offsets_from_stats, normalize_frame_offsets


def test_frame_offsets_are_ordered_lookbacks_and_legacy_defaults():
    assert normalize_frame_offsets([15, 0]) == (15, 0)
    assert frame_indices(4, [15, 0]) == (0, 4)
    assert frame_indices(20, [15, 0]) == (5, 20)
    assert frame_offsets_from_stats({}) == (0,)
    with pytest.raises(ValueError, match="end with 0"):
        normalize_frame_offsets([0, 15])


def test_dataset_temporal_views_duplicate_start_and_share_crop():
    frame = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    sequence = np.stack([frame, np.roll(frame, 1, axis=0)], axis=0)
    torch.manual_seed(7)
    eih, shoulder = prepare_temporal_views(
        sequence, sequence, timestep=0, frame_offsets=[15, 0], augment=True,
    )
    assert eih.shape == (2, 3, 128, 128)
    assert torch.equal(eih[0], eih[1])
    assert torch.equal(eih, shoulder)


def test_control_step_buffer_uses_exact_lookback_and_start_duplication():
    history = TemporalObservationBuffer([2, 0])
    history.reset({"shoulder": np.full((2, 2, 3), 10, dtype=np.uint8)})
    start = history.batch("shoulder", "cpu")
    assert start.shape == (1, 2, 3, 2, 2)
    assert torch.equal(start[:, 0], start[:, 1])
    history.append({"shoulder": np.full((2, 2, 3), 20, dtype=np.uint8)})
    history.append({"shoulder": np.full((2, 2, 3), 30, dtype=np.uint8)})
    current = history.batch("shoulder", "cpu")
    assert current[0, 0, 0, 0, 0].item() == pytest.approx(10 / 255.0)
    assert current[0, 1, 0, 0, 0].item() == pytest.approx(30 / 255.0)


def test_image_dit_encodes_two_frames_with_shared_backbone():
    model = ImageDiT(
        x_dim=7, d_model=32, n_heads=4, depth=1, dim_feedforward=64,
        horizon=3, num_cameras=1, frame_offsets=[15, 0],
    )
    shoulder = torch.rand(1, 2, 3, 128, 128)
    encoded = model.forward_encoder(None, shoulder)
    assert len(encoded) == 1
    assert encoded[0].shape == (1, model.tokens_per_camera * 2, 32)
    assert model.temporal_embed.weight.shape == (2, 32)
