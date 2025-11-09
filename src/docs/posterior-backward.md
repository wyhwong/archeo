Posteriors refer to the posterior probability distributions for each ancestral model parameter.

---

## Infer Posterior by Backward approach

After generating a prior, you may want to use it in a backward model to infer ancestral posterior distributions. As mentioned in the methodology, the backward approach conditions on the existing parameter estimation samples. We therefore will need to first prepare the mass and spin posterior samples from a parameter estimation run.

For example, if you have a JSON file containing the posterior samples, you can load it using pandas as follows:

```python
samples = pd.read_json("path/to/pe_samples.json")

mass_posterior = samples["mass_1_source"]
spin_posterior = samples["a_1"]
```

Then the next step to obtain the posteriors from the backward model could be simply done as follows:

```python

posterior = prior.to_posterior(
    mass_posterior=mass_posterior,
    spin_posterior=spin_posterior,
)
```

---

## Tolerance in Matching Prior and Posterior Samples

Due to the discrete nature of the prior samples, we will need a non-zero tolerance value to match the posterior samples with the prior samples.
For details, please see the section of **Practical Implementation** in [Araújo Álvarez et al., 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad90a9).
By default, the tolerance is set to 0.05 for spin and 1.0 for mass (in unit of solar mass).

To customize the tolerance values, you may specify them as follows:

```python
# NOTE: do this before calling to_posterior()
prior.update_tolerances(
    spin_tolerance=<float>, # dimensionless
    mass_tolerance=<float>, # in unit of solar mass
)
```
