from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("archeo")
except PackageNotFoundError:
    # Running from source tree (not installed)
    __version__ = "0.0.0"
