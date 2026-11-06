from __future__ import annotations

import torch

from sonnix.modules import CosSinNumericalEncoder

############################################
#     Tests for CosSinNumericalEncoder     #
############################################


def test_cos_sin_numerical_encoder_create_rand_frequency_above_min() -> None:
    module = CosSinNumericalEncoder.create("rand", 100, frequency_range=(0.2, 1.0), generator=42)
    assert module.frequency.min() >= 0.2


def test_cos_sin_numerical_encoder_create_rand_frequency_below_max() -> None:
    module = CosSinNumericalEncoder.create("rand", 100, frequency_range=(0.2, 1.0), generator=42)
    assert module.frequency.max() <= 1.0


def test_cos_sin_numerical_encoder_create_rand_value_range_frequency_positive() -> None:
    module = CosSinNumericalEncoder.create("rand", 100, value_range=(0.2, 1.0), generator=42)
    assert module.frequency.min() > 0.0


def test_cos_sin_numerical_encoder_create_rand_same_seed_same_frequency() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=42)
    module2 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=42)
    assert module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_same_seed_same_phase_shift() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=42)
    module2 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=42)
    assert module1.phase_shift.equal(module2.phase_shift)


def test_cos_sin_numerical_encoder_create_rand_different_seed_different_frequency() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=1)
    module2 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=2)
    assert not module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_generator_object_same_as_seed() -> None:
    module1 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0), generator=42)
    module2 = CosSinNumericalEncoder.create(
        "rand", 10, frequency_range=(0.2, 1.0), generator=torch.Generator().manual_seed(42)
    )
    assert module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_none_uses_global_generator() -> None:
    torch.manual_seed(42)
    module1 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0))
    torch.manual_seed(42)
    module2 = CosSinNumericalEncoder.create("rand", 10, frequency_range=(0.2, 1.0))
    assert module1.frequency.equal(module2.frequency)


def test_cos_sin_numerical_encoder_create_rand_forward_shape() -> None:
    module = CosSinNumericalEncoder.create("rand", 6, frequency_range=(0.2, 1.0), generator=42)
    assert module(torch.randn(4, 1)).shape == (4, 1, 12)
