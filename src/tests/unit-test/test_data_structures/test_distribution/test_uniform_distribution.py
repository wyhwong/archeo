import numpy as np

from archeo.data_structures.distribution import Uniform


SAMPLE_SIZE = 100000


def test_uniform_distribution():

    dist = Uniform(low=0, high=10)

    assert dist.min == 0
    assert dist.max == 10

    samples = dist.draw(size=SAMPLE_SIZE)
    assert len(samples) == SAMPLE_SIZE
    assert all(0 <= sample <= 10 for sample in samples)

    # Check expected mean
    expected_mean = (dist.low + dist.high) / 2
    assert np.isclose(np.mean(samples), expected_mean, atol=expected_mean * 0.1)

    # Separate the samples into 10 bins, the expected count in each bin should be around 100
    bins = np.linspace(dist.low, dist.high, num=11)
    counts, _ = np.histogram(samples, bins=bins)
    expected_count = len(samples) / 10
    assert all(np.isclose(count, expected_count, atol=expected_count * 0.1) for count in counts)
