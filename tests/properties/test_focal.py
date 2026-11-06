from __future__ import annotations

import torch
from hypothesis import given, strategies as st

from sonnix.functional import binary_focal_loss, binary_focal_loss_with_logits

####################################################
#     Property tests for the binary focal loss     #
####################################################


@given(
    data=st.lists(
        st.tuples(
            st.floats(min_value=-8.0, max_value=8.0),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        min_size=1,
        max_size=16,
    ),
    gamma=st.floats(min_value=0.0, max_value=4.0),
)
def test_binary_focal_loss_with_logits_equals_probs_after_sigmoid(
    data: list[tuple[float, float]], gamma: float
) -> None:
    logits = torch.tensor([x for x, _ in data])
    target = torch.tensor([t for _, t in data])
    assert binary_focal_loss_with_logits(logits, target, gamma=gamma).allclose(
        binary_focal_loss(logits.sigmoid(), target, gamma=gamma), rtol=1e-3, atol=1e-3
    )
