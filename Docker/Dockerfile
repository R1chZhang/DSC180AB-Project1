ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"


USER root

RUN conda create -n lrgb python=3.9


RUN conda install pytorch=1.9 torchvision torchaudio -c pytorch -c nvidia
RUN conda install pyg=2.0.2 -c pyg -c conda-forge
RUN conda install pandas scikit-learn matplotlib


RUN conda install openbabel fsspec rdkit -c conda-forge

RUN pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html

RUN pip install performer-pytorch
RUN pip install torchmetrics==0.7.2
RUN pip install ogb
RUN pip install wandb

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]
