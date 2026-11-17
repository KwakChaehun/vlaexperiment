import numpy as np
import torch

from vla_eval.direct_feature_prediction import FeaturePredictor, context


def test_forecast_context_cannot_read_uncommitted_future_actions():
    state = np.zeros(8, np.float32)
    actions = np.arange(112, dtype=np.float32).reshape(16, 7)
    reference = context(state, state, actions, 4)
    changed = actions.copy()
    changed[4:] = 10000
    np.testing.assert_array_equal(reference, context(state, state, changed, 4))
    changed[3, 0] += 1
    assert not np.array_equal(reference, context(state, state, changed, 4))


def test_zero_initialized_residual_preserves_cached_features():
    torch.manual_seed(3)
    basis = torch.linalg.qr(torch.randn(16, 4)).Q
    model = FeaturePredictor(basis, torch.zeros(145), torch.ones(145))
    current = torch.randn(2, 10, 16).to(torch.bfloat16)
    previous = torch.randn_like(current)
    predicted = model(current, previous, torch.randn(2, 145), torch.ones(2, 10))
    assert predicted.dtype == current.dtype
    assert torch.equal(predicted, current)
