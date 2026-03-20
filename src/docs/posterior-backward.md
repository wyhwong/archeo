Posteriors refer to the posterior probability distributions for each ancestral model parameter.

---

## Infer Posterior by Backward approach

After generating a prior, you may want to use it in a backward model to infer ancestral posterior distributions. As mentioned in the methodology, the backward approach conditions on the existing parameter estimation samples. We therefore will need to first prepare the mass and spin posterior samples from a parameter estimation run.

For example, if you have a JSON file containing the posterior samples, you can load it using pandas as follows:

```python
import pandas as pd

samples = pd.read_json("path/to/pe_samples.json")

mass_posterior = samples["mass_1_source"]
spin_posterior = samples["a_1"]
```

Then the next step to obtain the posteriors from the backward model could be simply done as follows:

```python
from archeo import infer_ancestral_posterior_distribution

df_inferred_samples = infer_ancestral_posterior_distribution(
    df_binaries=df_binaries,
    mass_posterior_samples=mass_posterior.tolist(),
    spin_posterior_samples=spin_posterior.tolist(),
    n_workers=-1,
)
```

---

## Tolerance in Matching Prior and Posterior Samples

Due to the discrete nature of the prior samples, we will need a non-zero tolerance value to match the posterior samples with the prior samples.
For details, please see the section of **Practical Implementation** in [Araújo Álvarez et al., 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9).
By default, the binsize is set to 0.1 for spin and 1.0 for mass (in unit of solar mass),
meaning the tolerance is 0.05 for spin and 1.0 for mass (in unit of solar mass), same as v1.

To customize the tolerance values, you may specify them as follows:

```python
df_inferred_samples = infer_ancestral_posterior_distribution(
    df_binaries=df_binaries,
    mass_posterior_samples=mass_posterior.tolist(),
    spin_posterior_samples=spin_posterior.tolist(),
    binsize_mass=<desired value>,
    binsize_spin=<desired value>,
    n_workers=-1,
)
```
