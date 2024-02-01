from glob import glob
from typing import Any

import numpy as np
import pandas as pd

import core
import env
import logger
import schemas
import services.posterior
import services.prior


np.random.seed(env.RANDOM_SEED)
local_logger = logger.get_logger(__name__)


class SimulationFacade:
    """Facade for the simulation."""

    def __init__(self, main_settings: dict[str, Any], prior_settings: dict[str, Any]) -> None:
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

        core.utils.check_and_create_dir(env.RESULTS_DIR)
        num_experiment = len(glob(f"{env.RESULTS_DIR}/*")) + 1
        self._output_dir = f"{env.RESULTS_DIR}/{num_experiment}"
        if self._main_settings["run_label"]:
            self._output_dir += "_" + self._main_settings["run_label"]
        core.utils.check_and_create_dir(self._output_dir)
        core.utils.save_as_yml(f"{self._output_dir}/main.yml", self._main_settings)
        core.utils.save_as_yml(f"{self._output_dir}/prior.yml", self._prior_settings)

    def _run_prior_simulation(self) -> None:
        """
        Run the prior simulation.
        """

        settings = schemas.binary.BinarySettings(
            is_spin_aligned=self._prior_settings["is_spin_aligned"],
            spin=schemas.common.Domain(
                low=self._prior_settings["spin"]["low"],
                high=self._prior_settings["spin"]["high"],
            ),
            mass=schemas.common.Domain(
                low=self._prior_settings["mass"]["low"],
                high=self._prior_settings["mass"]["high"],
            ),
            mass_ratio=schemas.common.Domain(
                low=self._prior_settings["mass_ratio"]["low"],
                high=self._prior_settings["mass_ratio"]["high"],
            ),
            phi=schemas.common.Domain(
                low=self._prior_settings["phi"]["low"] * np.pi,
                high=self._prior_settings["phi"]["high"] * np.pi,
            ),
            theta=schemas.common.Domain(
                low=self._prior_settings["theta"]["low"] * np.pi,
                high=self._prior_settings["theta"]["high"] * np.pi,
            ),
        )
        mass_from_pdf = (
            core.prior.mahapatra.get_mass_func_from_mahapatra(settings.mass)
            if self._prior_settings["mass"]["mahapatra"]
            else None
        )
        mass_ratio_from_pdf = (
            core.math.get_generator_from_csv(self._prior_settings["mass_ratio"]["data_path"])
            if self._prior_settings["mass_ratio"]["csv_path"]
            else None
        )
        fits = schemas.binary.Fits(self._prior_settings["fits"])
        services.prior.run_simulation(
            fits=fits,
            settings=settings,
            is_mass_injected=self._main_settings["prior"]["is_mass_injected"],
            num_binaries=self._prior_settings["num_binaries"],
            mass_ratio_from_pdf=mass_ratio_from_pdf,
            mass_from_pdf=mass_from_pdf,
            output_dir=self._output_dir,
        )

        local_logger.info("Finished running the prior simulation.")

    def _run_prior_visualization(self) -> None:
        """
        Visualize the prior.
        """

        df = pd.read_feather(f"{self._output_dir}/prior.feather")
        core.visualization.prior.distribution(df, output_dir=self._output_dir)
        core.visualization.prior.kick_against_spin(df, output_dir=self._output_dir)
        core.visualization.prior.kick_distribution_on_spin(df, output_dir=self._output_dir)

        local_logger.info("Finished visualizing the prior.")

    def _run_posterior_estimation(self) -> None:
        """
        Run the posterior estimation.
        """

        posteriors: dict[str, pd.DataFrame] = {}
        if self._main_settings["posterior"]["json_path"]:
            for label, filepath in self._main_settings["posterior"]["json_path"].items():
                posteriors[label] = core.posterior.sampler.get_posterior_from_json(filepath)

        if self._main_settings["posterior"]["h5_path"]:
            for label, filepath in self._main_settings["posterior"]["h5_path"].items():
                posteriors[label] = core.posterior.sampler.get_posterior_from_h5(filepath)

        df_prior = pd.read_feather(f"{self._output_dir}/prior.feather")
        sampler = core.posterior.sampler.PosteriorSampler(
            df=df_prior,
            is_mass_injected=self._main_settings["posterior"]["is_mass_injected"],
            n_sample=self._main_settings["posterior"]["n_sample"],
            spin_tolerance=self._main_settings["posterior"]["spin_tolerance"],
            mass_tolerance=self._main_settings["posterior"]["mass_tolerance"],
        )

        for label, posterior in posteriors.items():
            for bh_index in [1, 2]:
                posterior_label = f"{label},BH{bh_index}"
                df_posterior = services.posterior.infer_parental_posterior(
                    sampler=sampler,
                    label=posterior_label,
                    spin_posterior=posterior[f"a_{bh_index}"],
                    mass_posterior=posterior[f"mass_{bh_index}_source"],
                    output_dir=self._output_dir,
                )
                local_logger.info("Visualizing the posterior (%s)...", posterior_label)
                core.visualization.posterior.mass_estimates(
                    df_posterior,
                    posterior_label,
                    output_dir=self._output_dir,
                )
                core.visualization.posterior.corner_estimates(
                    dfs=[df_posterior],
                    labels=[posterior_label],
                    filename=f"{posterior_label}.png",
                    output_dir=self._output_dir,
                )
                core.visualization.posterior.cumulative_kick_probability_curve(
                    dfs=[df_posterior],
                    labels=[posterior_label],
                    filename=f"{posterior_label}.png",
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
