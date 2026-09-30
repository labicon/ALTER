import torch
import pytest

from src.image_coordination import (
    ALL_BLOCKS_DECODER_EXECUTION,
    DECODER_CONDITIONING_CROSS_ATTN,
    DECODER_CONDITIONING_POOLED,
    LEGACY_DECODER_EXECUTION,
    SIDE_NET_FUSION_CONCAT,
    ImageCoordinationHead,
    side_net_kwargs_from_stats,
)


def _run_and_count_blocks(decoder_execution):
    head = ImageCoordinationHead(
        base_d_model=32,
        d_model=16,
        n_heads=4,
        depth=4,
        dim_feedforward=32,
        horizon=5,
        num_cameras=1,
        tokens_per_camera=4,
        decoder_execution=decoder_execution,
    )
    calls = [0] * 4
    hooks = []
    for index, block in enumerate(head.F.decoder_blocks):
        hooks.append(block.register_forward_hook(
            lambda _module, _args, _output, i=index: calls.__setitem__(i, calls[i] + 1)
        ))

    x = torch.randn(2, 5, 7)
    sigma = torch.ones(2, 1, 1)
    encoder_outputs = [torch.randn(2, 4, 32) for _ in range(3)]
    head.forward_residual(x, sigma, encoder_outputs)

    for hook in hooks:
        hook.remove()
    return calls


def test_legacy_decoder_execution_only_runs_projected_layers():
    assert _run_and_count_blocks(LEGACY_DECODER_EXECUTION) == [1, 1, 0, 0]


def test_corrected_decoder_execution_runs_configured_depth():
    assert _run_and_count_blocks(ALL_BLOCKS_DECODER_EXECUTION) == [1, 1, 1, 1]


class _FakeBase:
    def __init__(self):
        self.F_ema = self
        self.null_token = torch.zeros(1, 4, 8)

    def sample_noise_distribution(self, batch_size, generator=None):
        return torch.ones(batch_size, 1, 1)

    def forward_encoder(self, _eih, shoulder):
        batch_size = shoulder.shape[0]
        return [torch.zeros(batch_size, 4, 8) for _ in range(3)]

    def _D_from_enc(self, x, _sigma, _enc, use_ema=True):
        return torch.zeros_like(x)

    def loss_weighting(self, sigma):
        return torch.ones_like(sigma)


def test_mixed_update_records_separate_losses_and_metadata(tmp_path):
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=3,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, decoder_execution=LEGACY_DECODER_EXECUTION,
    )
    batch = (
        None,
        torch.zeros(2, 3, 8, 8),
        torch.ones(2, 3, 7),
    )
    loss, grad, metrics = head.update_mixed(batch, batch, _FakeBase())
    assert loss >= 0.0
    assert grad >= 0.0
    assert set(metrics) == {
        "twoarm_loss", "singlearm_loss", "twoarm_residual_rms",
        "singlearm_residual_rms", "singlearm_weight",
    }

    head.checkpoint_metadata = {"dataset_manifest": {"twoarm_train_files": ["a.pkl"]}}
    path = tmp_path / "mixed.pt"
    head.save(path)
    saved = torch.load(path, weights_only=True)
    assert saved["decoder_execution"] == LEGACY_DECODER_EXECUTION
    assert saved["decoder_conditioning"] == DECODER_CONDITIONING_POOLED
    assert saved["metadata"]["last_train_metrics"] == metrics
    assert saved["metadata"]["training_metric_ema"] == metrics
    assert saved["metadata"]["dataset_manifest"]["twoarm_train_files"] == ["a.pkl"]


def test_side_net_add_preserves_legacy_context_shape():
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=2,
        tokens_per_camera=4, use_side_net=True,
        decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
    )
    seen = []
    hook = head.F.decoder_blocks[0].register_forward_hook(
        lambda _module, args, _output: seen.append(args[2].shape)
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 8, 8) for _ in range(3)]
    img = torch.randn(2, 3, 32, 32)
    out = head.forward_residual(
        x, sigma, enc, imgs_eih=img, imgs_shoulder=img,
    )
    hook.remove()
    assert out.shape == x.shape
    assert seen == [torch.Size([2, 8, 8])]


def test_side_net_highres_add_keeps_base_token_count():
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, use_side_net=True,
        side_net_input_size=256,
        decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 4, 8) for _ in range(3)]
    img = torch.randn(2, 3, 64, 64)
    out = head.forward_residual(x, sigma, enc, imgs_shoulder=img)
    assert out.shape == x.shape


def test_side_net_concat_appends_finer_side_stream():
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=2,
        tokens_per_camera=4, use_side_net=True,
        side_net_input_size=256, side_net_fusion=SIDE_NET_FUSION_CONCAT,
        side_net_tokens_per_camera=16,
        decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
    )
    seen = []
    hook = head.F.decoder_blocks[0].register_forward_hook(
        lambda _module, args, _output: seen.append(args[2].shape)
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 8, 8) for _ in range(3)]
    img = torch.randn(2, 3, 64, 64)
    out = head.forward_residual(
        x, sigma, enc, imgs_eih=img, imgs_shoulder=img,
    )
    hook.remove()
    assert out.shape == x.shape
    assert seen == [torch.Size([2, 40, 8])]


def test_side_net_stats_helper_uses_legacy_fallbacks():
    legacy = {"tokens_per_camera": 4, "use_side_net": True}
    assert side_net_kwargs_from_stats(legacy, base_tokens_per_camera=16) == {
        "tokens_per_camera": 4,
        "use_side_net": True,
        "side_net_input_size": 128,
        "side_net_fusion": "add",
        "side_net_tokens_per_camera": 4,
    }

    newer = {
        "tokens_per_camera": 4,
        "use_side_net": True,
        "side_net_input_size": 256,
        "side_net_fusion": "concat",
        "side_net_tokens_per_camera": 16,
    }
    assert side_net_kwargs_from_stats(newer, base_tokens_per_camera=4) == newer



def test_legacy_checkpoint_without_decoder_conditioning_loads(tmp_path):
    src = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, decoder_execution=LEGACY_DECODER_EXECUTION,
    )
    path = tmp_path / "legacy_no_conditioning.pt"
    torch.save({
        "model": src.F.state_dict(),
        "model_ema": src.F_ema.state_dict(),
        "decoder_execution": LEGACY_DECODER_EXECUTION,
    }, path)

    dst = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, decoder_execution=LEGACY_DECODER_EXECUTION,
    )
    assert dst.load(path)


def test_cross_attention_head_returns_action_shape():
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=2,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
        decoder_conditioning=DECODER_CONDITIONING_CROSS_ATTN,
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 4, 8) for _ in range(3)]
    out = head.forward_residual(x, sigma, enc)
    assert out.shape == x.shape


def test_side_net_concat_cross_attention_sees_all_tokens():
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=2,
        tokens_per_camera=4, use_side_net=True,
        side_net_fusion=SIDE_NET_FUSION_CONCAT,
        side_net_tokens_per_camera=16,
        decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
        decoder_conditioning=DECODER_CONDITIONING_CROSS_ATTN,
    )
    seen = []
    hook = head.F.decoder_blocks[0].cross_attn.register_forward_hook(
        lambda _module, args, _output: seen.append(args[1].shape)
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 8, 8) for _ in range(3)]
    img = torch.randn(2, 3, 32, 32)
    out = head.forward_residual(
        x, sigma, enc, imgs_eih=img, imgs_shoulder=img,
    )
    hook.remove()
    assert out.shape == x.shape
    assert seen == [torch.Size([2, 40, 8])]


def test_two_frame_side_stream_and_checkpoint_temporal_guard(tmp_path):
    head = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=8, use_side_net=True,
        side_net_fusion=SIDE_NET_FUSION_CONCAT,
        side_net_tokens_per_camera=4, frame_offsets=[15, 0],
        decoder_execution=ALL_BLOCKS_DECODER_EXECUTION,
    )
    x = torch.randn(2, 3, 7)
    sigma = torch.ones(2, 1, 1)
    enc = [torch.randn(2, 8, 8) for _ in range(3)]
    image = torch.randn(2, 2, 3, 32, 32)
    assert head.forward_residual(x, sigma, enc, imgs_shoulder=image).shape == x.shape
    path = tmp_path / "two_frame.pt"
    head.save(path)
    incompatible = ImageCoordinationHead(
        base_d_model=8, d_model=8, n_heads=2, depth=1,
        dim_feedforward=16, horizon=3, num_cameras=1,
        tokens_per_camera=4, frame_offsets=[0],
    )
    with pytest.raises(ValueError, match="frame offset mismatch"):
        incompatible.load(path)
