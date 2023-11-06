import glob
import pandas as pd
import numpy as np

import services
import utils
import schemas
import visualization
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Setup

    1. Create output directory.
    2. Set random seed.
    3. Save main config to yml file.
    """
    results_dir = "./results"
    utils.common.check_and_create_dir(dirpath=results_dir)

    main_config = utils.common.read_dict_from_yml("configs/main.yml")
    np.random.seed(seed=main_config["seed"])
    run_label = f"{int(len(glob.glob(f'{results_dir}/*')) + 1)}_{main_config['run_label']}"
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
        df_prior = pd.read_csv(main_config["prior"]["csv_path"])
    else:
        prior_args = utils.common.read_dict_from_yml("configs/prior.yml")
        utils.common.save_dict_as_yml(f"{output_dir}/prior.yml", prior_args)

        prior_config = schemas.binary.BinaryConfig.from_dict(prior_args["binary_config"])

        func_from_pdf = {}
        for para in ["mass", "mass_ratio"]:
            if prior_args["binary_config"][para]["csv_path"]:
                func_from_pdf[para] = utils.binary.get_generator_from_csv(prior_args["binary_config"][para]["csv_path"])
            else:
                func_from_pdf[para] = None

        df_prior = services.prior.run_simulation(
            prior_config,
            main_config["prior"]["mass_injection"],
            prior_args["num_binaries"],
            output_dir,
            func_from_pdf["mass"],
            func_from_pdf["mass_ratio"],
        )

    visualization.prior.plot_dist(df_prior, output_dir)
    visualization.prior.plot_kick_against_spin(df_prior, output_dir)
    visualization.prior.plot_kick_distribution_on_spin(df_prior, output_dir)

    """
    Posterior

    1. Load posterior from json or h5 file.
    2. Infer parental posterior.
    3. Visualize the posterior.
    """
    posteriors = {}
    if main_config["posterior"]["json_path"]:
        for label, posterior_filepath in main_config["posterior"]["json_path"].items():
            posteriors[label] = utils.posterior.read_posterior_from_json(posterior_filepath)

    if main_config["posterior"]["h5_path"]:
        for label, posterior_filepath in main_config["posterior"]["h5_path"].items():
            posteriors[label] = utils.posterior.read_posterior_from_h5(posterior_filepath)

    for label, posterior in posteriors.items():
        for bh_index in [1, 2]:
            posterior_label = f"{label},BH{bh_index}"
            df_posterior = services.posterior.infer_parental_posterior(
                df_prior,
                posterior_label,
                posterior[f"a_{bh_index}"],
                posterior[f"mass_{bh_index}_source"],
                main_config["posterior"]["mass_injection"],
                output_dir=output_dir,
            )
            visualization.posterior.plot_mass_estimates(df_posterior, posterior_label, output_dir)
            visualization.posterior.plot_corner(df_posterior, posterior_label, output_dir=output_dir)
            visualization.posterior.plot_cumulative_kick_probability_curve(
                df_posterior, posterior_label, output_dir=output_dir
            )


if __name__ == "__main__":
    main()
