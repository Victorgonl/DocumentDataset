# DocumentDataset
ARG PROJECT=DocumentDataset

# linux image
FROM python:latest

# working directory in the container
WORKDIR /$PROJECT

# copy the current directory contents into the container
COPY . /$PROJECT

# update and install with apt
RUN apt update
RUN apt install git wget -y

# install cmake (to build sentencepiece)
RUN apt install cmake -y

RUN pip install ipykernel ipywidgets

## install Python packages
RUN pip install -r $PROJECT/requirements.txt

## install DocumentDataset
RUN pip install -e .