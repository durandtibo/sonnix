from __future__ import annotations

import pytest
import torch

from sonnix.functional import binary_focal_loss, binary_focal_loss_with_logits

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")

####################################################
#     Property tests for the binary focal loss     #
####################################################


@hypothesis.given(
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
        binary_focal_loss(logits.sigmoid(), target, gamma=gamma), atol=1e-5
    )
