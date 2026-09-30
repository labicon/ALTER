from types import MethodType

import pytest
import torch

from src.image_diffusion import ImageConditional_ODE


def test_update_mixed_reduces_domains_separately_and_steps_once():
    model = ImageConditional_ODE.__new__(ImageConditional_ODE)
    model.F = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.F.weight.fill_(1.0)
    model.optim = torch.optim.SGD(model.F.parameters(), lr=0.1)

    def training_loss(self, scale, _imgs_eih, _imgs_shoulder):
        return self.F.weight.sum() * scale

    model._training_loss = MethodType(training_loss, model)
    ema_calls = []
    model.ema_update = lambda: ema_calls.append(True)

    multiarm_batch = (torch.tensor(2.0), None, None)
    singlearm_batch = (torch.tensor(3.0), None, None)
    loss, grad_norm, metrics = model.update_mixed(
        multiarm_batch,
        singlearm_batch,
        multiarm_weight=0.5,
        singlearm_weight=2.0,
    )

    assert loss == pytest.approx(7.0)
    assert grad_norm == pytest.approx(7.0)
    assert metrics == {
        "multiarm_loss": pytest.approx(2.0),
        "singlearm_loss": pytest.approx(3.0),
    }
    assert model.F.weight.item() == pytest.approx(0.3)
    assert ema_calls == [True]
