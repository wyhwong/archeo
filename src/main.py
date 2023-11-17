import glob
import pandas as pd
import numpy as np

import services
import utils
import schemas
import visualization
import warnings

warnings.filterwarnings("ignore")
logger = utils.logger.get_logger(logger_name="main")


def main() -> None:
    """
    Setup

    1. Create output directory.
    2. Set random seed.
    3. Save main config to yml file.
    """
    logger.info("Setting up...")
    results_dir = "./results"
    utils.common.check_and_create_dir(dirpath=results_dir)

    main_config = utils.common.read_dict_from_yml("configs/main.yml")
    np.random.seed(seed=main_config["seed"])
    run_label = (
        f"{int(len(glob.glob(f'{results_dir}/*')) + 1)}_{main_config['run_label']}"
    )
    output_dir = f"{results_dir}/{run_label}"
    utils.common.check_and_create_dir(output_dir)
    utils.common.save_dict_as_yml(f"{output_dir}/main.yml", main_config)

    """
    Prior
    
    1. Load prior from csv file if `load_results` is True.
    2. Else, run prior simulation and save the results to csv file.
    3. Visualize the prior.
    """
    if main_config["prior"]["load_results"]:
        logger.info("Loading prior from csv file...")
        df_prior = pd.read_csv(main_config["prior"]["csv_path"])
    else:
        logger.info("Running the prior simulation...")
        prior_args = utils.common.read_dict_from_yml("configs/prior.yml")
        utils.common.save_dict_as_yml(f"{output_dir}/prior.yml", prior_args)

        prior_config = schemas.binary.BinaryConfig.from_dict(
            prior_args["binary_config"]
        )

        func_from_pdf = {}
        for para in ["mass", "mass_ratio"]:
            if prior_args["binary_config"][para]["csv_path"]:
                func_from_pdf[para] = utils.binary.get_generator_from_csv(
                    prior_args["binary_config"][para]["csv_path"]
                )
            elif para == "mass" and prior_args["binary_config"][para]["mahapatra"]:
                func_from_pdf[para] = utils.mahapatra.get_mass_func_from_mahapatra(
                    prior_config.mass
                )
            else:
                func_from_pdf[para] = None

        df_prior = services.prior.run_simulation(
            config=prior_config,
            is_mass_injected=main_config["prior"]["is_mass_injected"],
            num_binaries=prior_args["num_binaries"],
            mass_ratio_from_pdf=func_from_pdf["mass_ratio"],
            mass_from_pdf=func_from_pdf["mass"],
            output_dir=output_dir,
        )

    logger.info("Visualizing the prior...")
    visualization.prior.distribution(df_prior, output_dir)
    visualization.prior.kick_against_spin(df_prior, output_dir)
    visualization.prior.kick_distribution_on_spin(df_prior, output_dir)

    """
    Posterior

    1. Load posterior from json or h5 file.
    2. Infer parental posterior.
    3. Visualize the posterior.
    """
    logger.info("Loading posterior from json or h5 file...")
    posteriors = {}
    if main_config["posterior"]["json_path"]:
        for label, filepath in main_config["posterior"]["json_path"].items():
            posteriors[label] = utils.posterior.read_posterior_from_json(filepath)

    if main_config["posterior"]["h5_path"]:
        for label, filepath in main_config["posterior"]["h5_path"].items():
            posteriors[label] = utils.posterior.read_posterior_from_h5(filepath)

    logger.info("Infering parental posterior...")
    sampler = utils.posterior.PosteriorSampler(
        df=df_prior,
        is_mass_injected=main_config["posterior"]["is_mass_injected"],
        n_sample=main_config["posterior"]["n_sample"],
        spin_tolerance=main_config["posterior"]["spin_tolerance"],
        mass_tolerance=main_config["posterior"]["mass_tolerance"],
    )
    for label, posterior in posteriors.items():
        for bh_index in [1, 2]:
            posterior_label = f"{label},BH{bh_index}"
            df_posterior = services.posterior.infer_parental_posterior(
                sampler=sampler,
                label=posterior_label,
                spin_posterior=posterior[f"a_{bh_index}"],
                mass_posterior=posterior[f"mass_{bh_index}_source"],
                output_dir=output_dir,
            )
            logger.info("Visualizing the posterior (%s)...", posterior_label)
            visualization.posterior.mass_estimates(
                df_posterior, posterior_label, output_dir
            )
            visualization.posterior.corner_estimates(
                df_posterior, posterior_label, output_dir=output_dir
            )
            visualization.posterior.cumulative_kick_probability_curve(
                df_posterior, posterior_label, output_dir=output_dir
            )


if __name__ == "__main__":
    main()
