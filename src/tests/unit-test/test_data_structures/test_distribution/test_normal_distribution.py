import numpy as np

from archeo.data_structures.distribution import Normal


SAMPLE_SIZE = 100000


def test_normal_distribution():

    expected_mean = 5
    expected_std = 10
    dist = Normal(mean=expected_mean, std=expected_std)

    assert dist.min == float("-inf")
    assert dist.max == float("inf")

    samples = dist.draw(size=SAMPLE_SIZE)
    assert len(samples) == SAMPLE_SIZE

    # Check expected mean and std
    assert np.isclose(np.mean(samples), dist.mean, atol=expected_mean * 0.1)
    assert np.isclose(np.std(samples), dist.std, atol=expected_std * 0.1)
