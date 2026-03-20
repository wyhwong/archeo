Priors refer to the prior probability distributions for each ancestral model parameter.

---

## Initializing Preset Priors

Initializing a preset prior in archeo is straightforward.
For example, to initialize generate an second generation aligned spin prior for testing, you can use the following code:

```python
from archeo import get_binary_generation_pipeline

pipeline = get_binary_generation_pipeline("2g_aligned_spin")
df_binaries, binary_generator = pipeline(size=5000, n_workers=-1)
# Note that the size means number of samples to be generated in the prior.
# And n_workers specifies the number of parallel workers to use for sample generation.
# Setting n_workers=-1 will use all available CPU cores for parallel processing.
```

Note that the prior initialization will take some time for:

1. Downloading the fits model if not already cached.
2. Generating samples from the fits model.

The wait time mainly depends on the number of samples specified in the prior configuration.
During the sample generation process, you will see progress bars indicating the status of sample generation.

If you have generated the prior samples once,
they could be cached locally for future use,
and subsequent initializations will be much faster.

```python
df_binaries.to_parquet("path/to/save/simulated_binaries.parquet")

# You can also store the binary generator configuration
import json

with open("path/to/save/binary_generator_config.json", "w", encoding="utf-8") as fp:
    json.dump(binary_generator.model_dump(), fp, indent=4)
```

You may replace the pipeline key `2g_aligned_spin` with any other supported preset prior name, here we list all available preset priors and corresponding configurations:

| Name | Fits Model | Spin Aligned | $\chi_1$ | $\chi_2$ | $\phi_1$ [rad] | $\phi_2$ [rad] | $\theta_1$ [rad] | $\theta_2$ [rad] | $m_1 [M_\odot]$ | $m_2 [M_\odot]$ | $q$ | Uniform in $q$ |
|------------------------------------|------------------|-----|-------|-------|------------|------------|-----------|-----------|------------------|-------------------|-------|----|
| agnostic_precession_spin           | NRSur7dq4Remnant | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ |     5 - 200      |      5 - 200      | 1 - 6 | ❌ |
| agnostic_aligned_spin              | NRSur3dq8Remnant | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ |     5 - 200      |      5 - 200      | 1 - 6 | ❌ |
| 2g_precession_spin                 | NRSur7dq4Remnant | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ |     5 - 65       |      5 - 65       | 1 - 6 | ❌ |
| 2g_aligned_spin                    | NRSur3dq8Remnant | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ |     5 - 65       |      5 - 65       | 1 - 6 | ❌ |
| ng_precession_spin                 | NRSur7dq4Remnant | ❌ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | Depends on input | Depends on input  | 1 - 6 | ❌ |
| ng_aligned_spin                    | NRSur3dq8Remnant | ✅ | 0 - 1 | 0 - 1 | 0 - $2\pi$ | 0 - $2\pi$ | 0 - $\pi$ | 0 - $\pi$ | Depends on input | Depends on input  | 1 - 6 | ❌ |

A special note on the `ng_precession_spin` and `ng_aligned_spin` priors: instead of sampling the black holes from defined distributions,
here we sample the black holes from a simulated population of binary black holes.
The process is illustrated in the following code snippet:

```python
import pandas as pd
from archeo import get_binary_generation_pipeline

# Here we first generate a population of 1g-1g binaries with aligned spins.
# We will then use the remnants of these 1g-1g binaries
# to generate 2g-1g binaries with precession spins.
pipeline = get_binary_generation_pipeline("2g_aligned_spin")
df_1g1g_aligned_spin_binaries, binary_generator = pipeline(size=5000, n_workers=-1)

# Here we use the pipeline of `ng_precession_spin`.
# When calling the pipeline, we specify df_bh1_binaries=df_1g1g_aligned_spin_binaries.
# This means we will sample one black hole from the remnant population,
# and one black hole from the first generation black hole distribution.
pipeline = get_binary_generation_pipeline("ng_precession_spin")
df_2g1g_binaries, binary_generator = pipeline(
    size=5000, n_workers=-1, df_bh1_binaries=df_1g1g_aligned_spin_binaries
)

# If you want to generate 2g-2g binaries,
# you can simply specify df_bh2_binaries=df_1g1g_aligned_spin_binaries
# when calling the `ng_precession_spin` pipeline.
# i.e.
# df_2g2g_binaries, binary_generator = pipeline(
#     size=5000,
#     n_workers=-1,
#     df_bh1_binaries=df_1g1g_aligned_spin_binaries,
#     df_bh2_binaries=df_1g1g_aligned_spin_binaries,
# )
```


---

## Customizing Simulation Configuration

You can also create a customized simulation using the data structure in archeo.
We will illustrate the process in the following code snippet:

```python
from archeo.data_structures.distribution import Normal, PiecewiseUniform, Uniform
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator
from archeo.data_structures.physics.mahapatra import MahapatraMassFunction
from archeo.data_structures.math import Domain
from archeo.simulation.simulate_merger import simulate_black_hole_mergers

bh_generator = BlackHoleGenerator(
    mass_distribution=MahapatraMassFunction(mass=Domain(5, 65)),
    # Here it means region with spin magnitude between 0 and 0.2 has 80% probability,
    # and region with spin magnitude between 0.2 and 1 has 20% probability.
    spin_magnitude_distribution=PiecewiseUniform(
        uniforms={
            Uniform(low=0, high=0.2): 0.8,
            Uniform(low=0.2, high=1.0): 0.2,
        }
    ),
    phi_distribution=Uniform(low=0, high=2 * np.pi),
    theta_distribution=Uniform(low=0, high=np.pi),
)
binary_generator = BinaryGenerator(
    primary_black_hole_source=bh_generator,
    secondary_black_hole_source=bh_generator,
    is_aligned_spin=False,
)
# Note that fits should be
fits = Fits.NRSUR7DQ4REMNANT if not is_aligned_spin else Fits.NRSUR3DQ8REMNANT
size = 5000
n_workers = -1
black_hole_mergers = simulate_black_hole_mergers(binary_generator, fits, size, n_workers)
```

To have more specific control over the prior configuration,
you may want to look into [archeo.data_structures](https://github.com/wyhwong/archeo/blob/main/src/archeo/data_structures/).

In the example, you may have noticed:
We also implemented an astrophysical motivated mass distribution prior based on [Mahapatra et al., 2022](https://arxiv.org/abs/2209.05766).
You can leverage this mass function to simulate more astrophysically realistic binary black hole populations.
