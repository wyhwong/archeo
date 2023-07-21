FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y tzdata python3 python3-pip -y
RUN pip3 install pyyaml p_tqdm scipy seaborn numpy pandas matplotlib jupyterlab notebook tables corner
RUN conda install surfinbh -c conda-forge
RUN jt -t monokai

ARG USERNAME
ARG USER_ID
ARG GROUP_ID
ARG TZ
ENV TZ=${TZ}
RUN groupadd --gid ${GROUP_ID} ${USERNAME} && \
    adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${USERNAME}

USER ${USERNAME}
