import archeo.core.utils
import archeo.env
import archeo.services.facade


def main():
    """Main function."""

    main_settings = archeo.core.utils.load_yml(archeo.env.CONFIG_PATH)
    prior_settings = archeo.core.utils.load_yml(archeo.env.PRIOR_CONFIG_PATH)
    facade = archeo.services.facade.SimulationFacade(main_settings, prior_settings)
    facade.run()


if __name__ == "__main__":
    main()
