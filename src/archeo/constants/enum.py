import enum
import os

import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


class Fits(enum.StrEnum):
    """Surrogate models for binary black hole merger simulations.

    Attributes:
        NRSur3dq8Remnant: non precessing BHs with mass ratio<=8, anti-/aligned spin <= 0.8.
        NRSur7dq4Remnant: precessing BHs with mass ratio<=4, generic spin <= 0.8.
        surfinBH7dq2: precessing BHs with mass ratio <= 2, generic spin <= 0.8.

    Details please refer to https://pypi.org/project/surfinBH/.
    """

    NRSUR3DQ8REMNANT = "NRSur3dq8Remnant"
    NRSUR7DQ4REMNANT = "NRSur7dq4Remnant"
    SURFINBH7DQ2 = "surfinBH7dq2"

    def load(self):
        """Load a surfinBH fits.

        Returns:
            fits (surfinBH.surfinBH.SurFinBH): The loaded fits.
        """

        import numpy as np  # pylint: disable=import-outside-toplevel
        import surfinBH  # pylint: disable=import-outside-toplevel

        # TODO: Remove this when SurfinBH upgraded to numpy 2.0
        np.string_ = np.bytes_  # Here we fix the numpy incompatibility issue in SurfinBH

        local_logger.info(
            "Loading surfinBH %s, description: %s.",
            self.value,
            surfinBH.fits_collection[self.value].desc,
        )

        try:
            return surfinBH.LoadFits(self.value)
        except (OSError, KeyError) as e:
            local_logger.error("Failed to load surfinBH %s: %s", self.value, str(e))
            self.clean_up_surfinbh_data()
            return self.load()

    @staticmethod
    def clean_up_surfinbh_data():
        """Clean up the surfinBH data directory.

        We clean up in two situations:
        1. KeyError: this seems to be a bug in surfinBH,
           when installing the latest version (1.2.6).
        2. OSError: this happens when we interrupt the download
           of the surfinBH data files.
        """

        import surfinBH  # pylint: disable=import-outside-toplevel

        # Remove all files in the data directory
        data_dir = f"{os.path.dirname(surfinBH.__file__)}/data"

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                local_logger.warning(
                    "Cleaning up surfinBH data directory: removing %s due to error.",
                    f"{data_dir}/{file}",
                )
                os.remove(f"{data_dir}/{file}")
