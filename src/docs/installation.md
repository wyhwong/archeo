Our package archeo is available via [PyPI](https://pypi.org/project/archeo/). The installation is as simple as:

=== "pip"

    ```bash
    pip install archeo
    ```

=== "uv"

    ```bash
    uv add archeo
    ```

Most of archeo's functionality relies on:

* [`pandas`](https://pypi.org/project/pandas/): Core data manipulation library.
* [`surfinbh`](https://pypi.org/project/surfinbh/): Surrogate final Black Hole properties for mergers of binary black holes.
* [`corner`](https://pypi.org/project/corner/): Corner plot visualization library.

If you've got Python 3.11+ and `pip` installed, you're good to go.

## Optional dependencies

Archeo has the following optional dependency group:

* `archeo[ui]`: Web UI features powered by [Streamlit](https://streamlit.io/)

To install optional dependencies along with archeo, use one of the following commands:

=== "pip"

    ```bash
    # with the `ui` extra:
    pip install 'archeo[ui]'
    ```

=== "uv"

    ```bash
    # with the `ui` extra:
    uv add 'archeo[ui]'
    ```

## Install from repository

And if you prefer to install archeo directly from the repository:

=== "pip"

    ```bash
    pip install 'git+https://github.com/wyhwong/archeo@main'
    # or with UI features:
    pip install 'git+https://github.com/wyhwong/archeo@main#egg=archeo[ui]'
    ```

=== "uv"

    ```bash
    uv add 'git+https://github.com/wyhwong/archeo@main'
    # or with UI features:
    uv add 'git+https://github.com/wyhwong/archeo@main#egg=archeo[ui]'
    ```
