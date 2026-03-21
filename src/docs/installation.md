Our package archeo is available via [PyPI](https://pypi.org/project/archeo/). The installation is as simple as:

=== "pip"

    ```bash
    pip install archeo
    ```

=== "uv"

    ```bash
    uv add archeo
    ```

archeo's functionalities are built heavily on the following dependencies:

* [`pandas`](https://pypi.org/project/pandas/): Core data manipulation library.
* [`pydantic`](https://pypi.org/project/pydantic/): Data validation and settings management using Python type annotations.
* [`surfinbh`](https://pypi.org/project/surfinbh/): Surrogate final Black Hole properties for mergers of binary black holes.

If you've got Python 3.11+ and `pip` installed, you should be good to go.
We strongly recommend uv for its runtime performance.

## Install from repository

And if you prefer to install archeo directly from the repository:

=== "pip"

    ```bash
    pip install 'git+https://github.com/wyhwong/archeo@main'
    ```

=== "uv"

    ```bash
    uv add 'git+https://github.com/wyhwong/archeo@main'
    ```
