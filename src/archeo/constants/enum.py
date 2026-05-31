import enum
import os

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


class Fits(enum.StrEnum):
    """Enumeration of supported surfinBH fit models.

    Provides resilient lazy-loading helpers with retry and cache-cleanup utilities
    for corrupted model downloads.

    Attributes:
        NRSur3dq8Remnant: non precession BHs with mass ratio<=8, anti-/aligned spin <= 0.8.
        NRSur7dq4Remnant: precession BHs with mass ratio<=4, generic spin <= 0.8.
        surfinBH7dq2: precession BHs with mass ratio <= 2, generic spin <= 0.8.

    Details please refer to https://pypi.org/project/surfinBH/.
    """

    NRSUR3DQ8REMNANT = "NRSur3dq8Remnant"
    NRSUR7DQ4REMNANT = "NRSur7dq4Remnant"
    SURFINBH7DQ2 = "surfinBH7dq2"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(RuntimeError),
    )
    def load(self):
        """Load and return the configured surfinBH fit model.

        Returns:
            Any: Loaded surfinBH fit object.

        Notes:
            Applies retry logic and a NumPy compatibility workaround before loading.
        """

        import numpy as np  # pylint: disable=import-outside-toplevel
        import surfinBH  # pylint: disable=import-outside-toplevel

        # TODO: Remove this when SurfinBH upgraded to numpy 2.0
        np.string_ = np.bytes_  # Here we fix the numpy incompatibility issue in SurfinBH

        LOGGER.info(
            "Loading surfinBH %s, description: %s.",
            self.value,
            surfinBH.fits_collection[self.value].desc,
        )

        try:
            # NOTE: We added a retry logic because the model loading can fail
            # due to IO limitation of h5py. This happens when we have multiple
            # processes trying to load the same model at the same time.
            return surfinBH.LoadFits(self.value)
        except (OSError, KeyError) as e:
            LOGGER.error("Failed to load surfinBH %s: %s", self.value, str(e))
            self.clean_up_surfinbh_data()
            return self.load()

    @staticmethod
    def clean_up_surfinbh_data():
        """Remove cached surfinBH data files to recover from corrupted downloads.

        Returns:
            None
        """

        import surfinBH  # pylint: disable=import-outside-toplevel

        # Remove all files in the data directory
        data_dir = f"{os.path.dirname(surfinBH.__file__)}/data"

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                LOGGER.warning(
                    "Cleaning up surfinBH data directory: removing %s due to error.",
                    f"{data_dir}/{file}",
                )
                os.remove(f"{data_dir}/{file}")
