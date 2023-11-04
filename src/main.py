import glob
import pandas as pd
import numpy as np

import services
import utils
import schemas
import visualization


def main() -> None:
    """
    Main function.
    """
    results_dir = "./results"
    utils.common.check_and_create_dir(dirpath=results_dir)

    main_config = utils.common.read_dict_from_yml("configs/main.yml")
    np.random.seed(seed=main_config["seed"])
    run_label = f"{int(len(glob.glob(f'{results_dir}/*')) + 1)}_{main_config['run_label']}"
    output_dir = f"{results_dir}/{run_label}"
    utils.common.check_and_create_dir(output_dir)
    utils.common.save_dict_as_yml(f"{output_dir}/main.yml", main_config)

    if main_config["prior"]["load_results"]:
        df_prior = pd.read_csv(main_config["prior"]["path_to_csv"])
    else:
        prior_args = utils.common.read_dict_from_yml("configs/prior.yml")
        utils.common.save_dict_as_yml(f"{output_dir}/prior.yml", prior_args)

        prior_config = schemas.binary.BinaryConfig.from_dict(prior_args["binary_config"])
        prior_generator = utils.binary.BinaryGenerator(prior_config)
        df_prior = services.prior.run_prior_simulation(prior_generator, prior_args["num_binaries"], output_dir)

    visualization.prior.plot_prior_dist(df_prior, output_dir)


if __name__ == "__main__":
    main()
