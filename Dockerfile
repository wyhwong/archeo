FROM python:3.11-slim-buster

# Install dependencies
RUN apt update && apt install build-essential -y
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install gwsurrogate==0.5.0 surfinbh==1.2.0 pyyaml==6.0.1 scipy==1.11.1 \
    seaborn==0.12.2 numpy==1.25.1 pandas==2.0.3 matplotlib==3.7.2 tables==3.8.0 \
    corner==2.2.2 jupyterlab==4.0.3 notebook==7.0.0

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
COPY src /home/${USERNAME}/src
WORKDIR /home/${USERNAME}/src
