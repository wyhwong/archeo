def test_import():
    """Test that the package can be imported"""

    import archeo  # pylint: disable=import-outside-toplevel

    assert archeo is not None
