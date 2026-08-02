FROM nvcr.io/nvidia/dgl:24.04-py3

RUN apt update && apt install -y libxrender1 libxtst6 libxi6

WORKDIR /workspace
COPY ./ ./

RUN pip install -e .
