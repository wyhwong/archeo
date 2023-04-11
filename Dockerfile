FROM continuumio/miniconda3:latest
ENV TZ=Asia/Hong_Kong

RUN apt-get update && apt-get install -y tzdata python3 python3-pip -y
RUN pip3 install pyyaml p_tqdm scipy seaborn numpy pandas matplotlib jupyterthemes notebook tables
RUN conda install surfinbh -c conda-forge
RUN jt -t monokai
