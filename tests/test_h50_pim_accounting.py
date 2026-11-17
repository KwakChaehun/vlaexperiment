"""Guard batch, partial-sum, repeated-flow and per-call link accounting."""
import pytest
from scripts.summarize_h50_pim import account, physical_key


def test_full_batch_and_partial_sum_transfer():
    op=dict(vectors=51,input_dim=1536,output_dim=6144,module='model.ff.net.0')
    key=physical_key(op)
    assert key==(51,8192,1536)
    sim=dict(vectors=51,padded_output_dim=8192,padded_input_dim=1536,
        compute_ms=.5,readback_ms=.01,padded_input_bytes=51*1536*2,
        partial_output_bytes=51*8192*16*2)
    conversion={(51,1536,6144):.02}
    one,_=account({'operators':[op]}, {key:sim},conversion,.1,32,10)
    ten,_=account({'operators':[op]*10},{key:sim},conversion,1,32,10)
    assert ten['pim_compute_ms']==5  # Full 51-vector timing, not another x51.
    assert ten['linear_calls']==10
    assert ten['link_fixed_ms']==pytest.approx(.1)
    assert one['link_bytes']==51*1536*2+51*8192*16*2
    assert ten['hybrid_expert_estimate_ms']==pytest.approx(10*one['hybrid_expert_estimate_ms'])


def test_invalid_link_assumptions():
    with pytest.raises(ValueError):account({'operators':[]},{},{},0,0,0)
    with pytest.raises(ValueError):account({'operators':[]},{},{},0,32,-1)
