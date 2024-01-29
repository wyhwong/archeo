import schemas.common


def is_in_bounds(value: float, domain: schemas.common.Domain) -> bool:
    """
    Check if the value is in the domain.

    Args:
    -----
        value (float):
            The value to check.

        domain (schemas.common.Domain):
            The domain to check the value against.

    Returns:
    -----
        is_in_bound (bool):
            True if the value is in the domain, False otherwise.
    """

    in_bound = True

    if domain.low and value < domain.low:
        in_bound = False

    if domain.high and value > domain.high:
        in_bound = False

    return in_bound
