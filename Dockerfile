# DocumentDataset
ARG PROJECT=DocumentDataset

# linux image
FROM ubuntu:latest

# working directory in the container
WORKDIR /$PROJECT

# copy the current directory contents into the container
COPY . /$PROJECT

# install some apt packages
RUN apt update
RUN apt install git wget -y

# install cmake (to build sentencepiece)
RUN apt install cmake -y

# install Python and pip
RUN apt install python3 python3-pip -y

## install Python packages
RUN pip install -r $PROJECT/requirements.txt
