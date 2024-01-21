import services.facade
import core.utils
import env


def main():
    """Main function."""
    main_settings = core.utils.load_yml(env.CONFIG_PATH)
    prior_settings = core.utils.load_yml(env.PRIOR_CONFIG_PATH)
    facade = services.facade.SimulationFacade(main_settings, prior_settings)
    facade.run()


if __name__ == "__main__":
    main()
