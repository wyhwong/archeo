# STAGE 1
FROM python:3.11-slim-buster AS base

# Convert poetry.lock to requirements.txt
RUN pip3 install poetry
COPY ./pyproject.toml ./poetry.lock /
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes --without dev

# STAGE 2
FROM python:3.11-slim-buster AS last

# Extra dependencies for this project
RUN apt update && apt install build-essential -y

# Install dependencies from requirements.txt
COPY --from=base /requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Create data directory for surfinBH
RUN mkdir -p /usr/local/lib/python3.11/site-packages/surfinBH/data && \
    chmod 777 -R /usr/local/lib/python3.11/site-packages/surfinBH/data

# Create user
ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}
USER ${USERNAME}

# Set working directory and freeze scripts
COPY ./src /home/${USERNAME}/src
WORKDIR /home/${USERNAME}/src

CMD ["python3", "main.py"]