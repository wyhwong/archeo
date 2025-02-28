import subprocess

import archeo


# Run the Streamlit app
dist_path = archeo.__path__[0]
subprocess.run(["streamlit", "run", f"{dist_path}/frontend/app.py"])  # pylint: disable=subprocess-run-check
