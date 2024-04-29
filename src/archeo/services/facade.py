from glob import glob
from typing import Any

import numpy as np
import pandas as pd

import archeo.core
import archeo.env
import archeo.logger
import archeo.schemas
import archeo.services.posterior
import archeo.services.prior


np.random.seed(archeo.env.RANDOM_SEED)
local_logger = archeo.logger.get_logger(__name__)


class SimulationFacade:
    """Facade for the simulation."""

    def __init__(
        self,
        main_settings: dict[str, Any],
        prior_settings: dict[str, Any],
    ) -> None:
        """
        Args:
        -----
            main_settings (dict[str, Any]):
                The main settings.

            prior_settings (dict[str, Any]):
                The prior settings.

        Returns:
        -----
            None
        """

        self._main_settings = main_settings
        self._prior_settings = prior_settings
        self._output_dir: str = ""

        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize the simulation.

        1. Create output directory.
        2. Save main and prior settings.
        """

        archeo.core.utils.check_and_create_dir(archeo.env.RESULTS_DIR)
        num_experiment = len(glob(f"{archeo.env.RESULTS_DIR}/*")) + 1
        self._output_dir = f"{archeo.env.RESULTS_DIR}/{num_experiment}"
        if self._main_settings["run_label"]:
            self._output_dir += "_" + self._main_settings["run_label"]
        archeo.core.utils.check_and_create_dir(self._output_dir)
        archeo.core.utils.save_as_yml(f"{self._output_dir}/main.yml", self._main_settings)
        archeo.core.utils.save_as_yml(f"{self._output_dir}/prior.yml", self._prior_settings)

    def _run_prior_simulation(self) -> None:
        """
        Run the prior simulation.
        """

        settings = archeo.schemas.binary.BinarySettings(
            is_spin_aligned=self._prior_settings["spin"]["is_spin_aligned"],
            only_up_aligned_spin=self._prior_settings["spin"]["only_up_aligned_spin"],
            spin=archeo.schemas.common.Domain(
                low=self._prior_settings["spin"]["low"],
                high=self._prior_settings["spin"]["high"],
            ),
            mass=archeo.schemas.common.Domain(
                low=self._prior_settings["mass"]["low"],
                high=self._prior_settings["mass"]["high"],
            ),
            is_mahapatra=self._prior_settings["mass"]["is_mahapatra"],
            mass_ratio=archeo.schemas.common.Domain(
                low=self._prior_settings["mass_ratio"]["low"],
                high=self._prior_settings["mass_ratio"]["high"],
            ),
            phi=archeo.schemas.common.Domain(
                low=self._prior_settings["phi"]["low"] * np.pi,
                high=self._prior_settings["phi"]["high"] * np.pi,
            ),
            theta=archeo.schemas.common.Domain(
                low=self._prior_settings["theta"]["low"] * np.pi,
                high=self._prior_settings["theta"]["high"] * np.pi,
            ),
        )
        fits = archeo.schemas.binary.Fits(self._prior_settings["fits"])
        archeo.services.prior.run_simulation(
            fits=fits,
            settings=settings,
            is_mass_injected=self._main_settings["prior"]["is_mass_injected"],
            num_binaries=self._prior_settings["num_binaries"],
            output_dir=self._output_dir,
        )

        local_logger.info("Finished running the prior simulation.")

    def _run_prior_visualization(self) -> None:
        """
        Visualize the prior.
        """

        df = pd.read_feather(f"{self._output_dir}/prior.feather")
        archeo.core.visualization.prior.distribution(df, output_dir=self._output_dir)
        archeo.core.visualization.prior.kick_against_spin(df, output_dir=self._output_dir)
        archeo.core.visualization.prior.kick_distribution_on_spin(df, output_dir=self._output_dir)

        local_logger.info("Finished visualizing the prior.")

    def _run_posterior_estimation(self) -> None:
        """
        Run the posterior estimation.
        """

        posteriors: dict[str, pd.DataFrame] = {}
        if self._main_settings["posterior"]["json_path"]:
            for label, filepath in self._main_settings["posterior"]["json_path"].items():
                posteriors[label] = archeo.core.posterior.sampler.get_posterior_from_json(filepath)

        if self._main_settings["posterior"]["h5_path"]:
            for label, filepath in self._main_settings["posterior"]["h5_path"].items():
                posteriors[label] = archeo.core.posterior.sampler.get_posterior_from_h5(filepath)

        df_prior = pd.read_feather(f"{self._output_dir}/prior.feather")
        sampler = archeo.core.posterior.sampler.PosteriorSampler(
            df=df_prior,
            is_mass_injected=self._main_settings["posterior"]["is_mass_injected"],
            n_sample=self._main_settings["posterior"]["n_sample"],
            spin_tolerance=self._main_settings["posterior"]["spin_tolerance"],
            mass_tolerance=self._main_settings["posterior"]["mass_tolerance"],
        )

        dfs_posterior = []
        labels_posterior = []
        for bh in [1, 2]:
            for label, posterior in posteriors.items():
                label_posterior = f"{label},BH{bh}"
                df_posterior = archeo.services.posterior.infer_parental_posterior(
                    sampler=sampler,
                    label=label_posterior,
                    spin_posterior=posterior[f"a_{bh}"],
                    mass_posterior=posterior[f"mass_{bh}_source"],
                    output_dir=self._output_dir,
                )

                dfs_posterior.append(df_posterior)
                labels_posterior.append(label_posterior)

                local_logger.info("Visualizing the posterior (%s)...", label_posterior)
                archeo.core.visualization.posterior.mass_estimates(
                    df=df_posterior,
                    label=label_posterior,
                    filename=f"{label_posterior}_mass_estimates.png",
                    output_dir=self._output_dir,
                )
                # NOTE: Here df_prior is not included because
                #       the prior may not be a mass-injected prior.
                archeo.core.visualization.posterior.corner_estimates(
                    dfs=[df_prior, df_posterior],
                    labels=["Prior", label_posterior],
                    filename=f"{label_posterior}_corner.png",
                    output_dir=self._output_dir,
                )
                archeo.core.visualization.posterior.second_generation_probability_curve(
                    dfs=[df_posterior],
                    labels=[label_posterior],
                    filename=f"{label_posterior}_kick_curve.png",
                    output_dir=self._output_dir,
                )
                archeo.core.visualization.posterior.effective_spin_estimates(
                    dfs=[df_prior, df_posterior],
                    labels=["Prior", label_posterior],
                    filename=f"{label_posterior}_effective_spin.png",
                    output_dir=self._output_dir,
                )
                archeo.core.visualization.posterior.precession_spin_estimates(
                    dfs=[df_prior, df_posterior],
                    labels=["Prior", label_posterior],
                    filename=f"{label_posterior}_precession_spin.png",
                    output_dir=self._output_dir,
                )

        archeo.core.visualization.posterior.second_generation_probability_curve(
            dfs=dfs_posterior,
            labels=labels_posterior,
            filename="kick_curves.png",
            output_dir=self._output_dir,
        )
        archeo.core.visualization.posterior.effective_spin_estimates(
            dfs=[df_prior] + dfs_posterior,
            labels=["Prior"] + labels_posterior,
            filename="effective_spins.png",
            output_dir=self._output_dir,
        )
        archeo.core.visualization.posterior.precession_spin_estimates(
            dfs=[df_prior] + dfs_posterior,
            labels=["Prior"] + labels_posterior,
            filename="precession_spins.png",
            output_dir=self._output_dir,
        )
        archeo.core.visualization.posterior.table_estimates(
            dfs=[df_prior] + dfs_posterior,
            labels=["Prior"] + labels_posterior,
            filename="table_estimates.png",
            output_dir=self._output_dir,
        )

        local_logger.info("Finished running the posterior estimation.")

    def run(self) -> None:
        """
        Run the simulation.
        """

        if self._main_settings["prior"]["load_results"]:
            local_logger.info("Loading prior from data file %s...", self._main_settings["prior"]["data_path"])
            df = pd.read_feather(self._main_settings["prior"]["data_path"])
            df.to_feather(f"{self._output_dir}/prior.feather")
        else:
            local_logger.info("Running the prior simulation...")
            self._run_prior_simulation()

        local_logger.info("Saving the prior to feather file...")

        local_logger.info("Visualizing the prior...")
        self._run_prior_visualization()

        local_logger.info("Running the posterior estimation...")
        self._run_posterior_estimation()

        local_logger.info("Facade finished running.")
