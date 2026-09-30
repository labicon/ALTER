import numpy as np
import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from src.coordination_history import history_indices
from hardware_training.current_frame import require_current_frame_hardware
from src.image_coordination import ImageCoordinationHead


def make_head():
    return ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=2,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, use_side_net=True, side_net_fusion="concat",
        side_net_tokens_per_camera=4, frame_offsets=(0,),
    )


def test_raw_time_not_pruned_index_and_no_future():
    timestamps = np.arange(41, dtype=np.int64) * 50_000_000
    assert history_indices(timestamps, 30, (1, .5, 0)).tolist() == [10, 20, 30]
    assert history_indices(timestamps, 3, (1, .5, 0)).tolist() == [0, 0, 3]
    timestamps[10] += 10_000_000
    assert history_indices(timestamps, 30, (1, .5, 0)).tolist() == [9, 20, 30]


def test_history_resets_at_gaps():
    assert history_indices(np.array([1, 2_000_000_001]), 1, (1, 0)).tolist() == [1, 1]


@pytest.mark.parametrize("legacy_side_metadata", [False, True])
def test_single_frame_checkpoint_remains_loadable(tmp_path, legacy_side_metadata):
    head = make_head()
    path = tmp_path / "current.pt"
    head.save(path)
    payload = torch.load(path, weights_only=True)
    assert "side_frame_offsets" not in payload
    if legacy_side_metadata:
        payload["side_frame_offsets"] = [0]
    torch.save(payload, path)
    restored = make_head()
    assert restored.load(path)
    for name, value in head.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_retired_history_checkpoint_rejected_before_weights_load(tmp_path):
    path = tmp_path / "retired.pt"
    # Even a partial legacy checkpoint must fail with the retirement message,
    # rather than falling through to permissive state-dict loading.
    torch.save({"frame_offsets": [0], "side_frame_offsets": [20, 10, 0]}, path)
    with pytest.raises(ValueError, match="camera history is retired"):
        make_head().load(path)


@pytest.mark.parametrize("stats", [
    {}, {"variant": "A", "frame_offsets": [0], "side_frame_offsets": [0],
         "head_history_seconds": [0.]},
])
def test_current_frame_hardware_stats_accepted(stats):
    require_current_frame_hardware(stats)


@pytest.mark.parametrize("stats", [
    {"variant": "B"}, {"frame_offsets": [20, 10, 0]},
    {"side_frame_offsets": [20, 10, 0]}, {"head_history_seconds": [1., .5, 0.]},
])
def test_history_hardware_stats_rejected(stats):
    with pytest.raises(ValueError, match="variant B is retired"):
        require_current_frame_hardware(stats)


def test_current_frame_noop_loss_preserves_diagnostics():
    class Base:
        def __init__(self):
            self.F_ema = self
            self.null_token = torch.zeros(1, 4, 8)

        def sample_noise_distribution(self, batch):
            return torch.ones(batch, 1, 1)

        def forward_encoder(self, eih, current):
            assert current.ndim == 4
            return [torch.zeros(len(current), 4, 8)] * 3

        def _D_from_enc(self, noisy, sigma, enc, use_ema=True):
            return torch.full_like(noisy, 17)

        def loss_weighting(self, sigma):
            return torch.ones_like(sigma)

    head = make_head()
    head.F.final_layer.linear.bias.data.fill_(.5)
    diagnostics = {}
    loss, residual = head._training_domain_loss(
        torch.ones(2, 3, 7), None, torch.zeros(2, 3, 128, 128), Base(),
        zero_residual_target=True,
        diagnostics=diagnostics,
    )
    assert torch.allclose(loss, residual.square().mean())
    assert diagnostics["base_error"].tolist() == [256, 256]


def test_node_and_offline_sampler_match_without_ros():
    from hardware_training.bench_failure_states import sample_chunk

    tree = ast.parse((Path(__file__).parents[1] / "ros2_nodes/xarm_codiff_coordination_inference_node.py").read_text())
    node_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in node_class.body if isinstance(node, ast.FunctionDef) and node.name == "_sample")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "node_sample", "exec"), namespace)

    class Base:
        def __init__(self):
            self.F_ema = self
            self.null_token = torch.zeros(1, 4, 8)
            self.N = 2
            self.device = torch.device("cpu")
            self.x_dim = 7
            self.sigma_s = torch.tensor([1., .5])
            self.scale_s = torch.ones(2)
            self.coeff1 = torch.ones(2)
            self.coeff2 = torch.ones(2)
            self.t_s = torch.tensor([1., .5, 0.])

        def forward_encoder(self, eih, image):
            assert image.ndim == 4
            return [torch.ones(len(image), 4, 8)] * 3

        def _D_from_enc(self, noisy, sigma, enc):
            return noisy * .5

    base = Base()
    head = make_head()
    head.F_ema.final_layer.linear.bias.data.fill_(.1)
    current = torch.rand(1, 3, 128, 128)
    anchor = np.zeros(7, dtype=np.float32)
    node = SimpleNamespace(base_model=base, coord_head=head, horizon=3, cfg_w=1.2, log_residual=False)
    for anchor_dims in (6, 7):
        torch.manual_seed(9)
        expected = sample_chunk(base, head, current, anchor, 1.2, base.device, 3, anchor_dims)
        torch.manual_seed(9)
        with torch.no_grad():
            actual = namespace["_sample"](node, current, anchor, anchor_dims)[0].numpy()
        assert np.array_equal(actual, expected)
