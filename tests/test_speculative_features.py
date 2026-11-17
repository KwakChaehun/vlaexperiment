"""The sparse path must actually shrink MLP work and preserve the prior bank."""
from types import SimpleNamespace
import torch
from vla_eval.speculative_features import PredictedMLPCache


def make_cache():
    class CountMLP(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rows=[]
        def forward(self,x):
            self.rows.append(x.numel()//x.shape[-1])
            return x.square()+2*x
    eagle=torch.nn.Module()
    eagle.vision_model=torch.nn.Module()
    eagle.vision_model.block=torch.nn.Module()
    eagle.vision_model.block.mlp=CountMLP()
    cache=PredictedMLPCache(SimpleNamespace(torch=torch,eagle=eagle))
    return cache,eagle.vision_model.block.mlp


def test_prediction_residual_selects_changed_token_and_reduces_rows():
    cache,mlp=make_cache();old=torch.arange(16.).reshape(1,4,4)
    cache.mode='capture';cache.bank={};expected=mlp(old)
    bank=cache.bank;name=next(iter(bank));saved=[x.clone() for x in bank[name]]
    actual=old.clone();actual[:,2]+=100
    cache.mode='correct';cache.ratio=.25
    result=mlp(actual)
    assert mlp.rows==[4,1]
    desired=expected.clone();desired[:,2]=actual[:,2].square()+2*actual[:,2]
    torch.testing.assert_close(result,desired)
    for before,after in zip(saved,bank[name]):assert torch.equal(before,after)
    cache.close()


def test_zero_reuses_prediction_and_all_updates_match_dense():
    cache,mlp=make_cache();old=torch.zeros(1,8,3);actual=torch.ones_like(old)
    cache.mode='capture';cache.bank={};cached=mlp(old)
    cache.mode='correct';cache.ratio=0
    assert torch.equal(mlp(actual),cached)
    assert mlp.rows==[8]
    cache.ratio=1
    assert torch.equal(mlp(actual),actual.square()+2*actual)
    assert mlp.rows==[8,8]
    cache.close()


def test_shape_change_is_rejected_instead_of_misaligned_reuse():
    import pytest
    cache,mlp=make_cache();cache.mode='capture';cache.bank={};mlp(torch.zeros(1,4,3))
    cache.mode='correct';cache.ratio=.25
    with pytest.raises(ValueError,match='token shape differs'):mlp(torch.zeros(1,5,3))
    cache.close()
