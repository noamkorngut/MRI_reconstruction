FROM ubuntu:18.04 as base

RUN apt update
# RUN apt upgrade -y

# Install python 3.7
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

RUN apt install python3-pip -y
RUN echo "dsdsd"
RUN apt install wget -y
RUN wget https://raw.githubusercontent.com/noamkorngut/MRI_reconstruction/master/requirements.txt
RUN python3 -m pip install -r requirements.txt 
RUN git clone https://github.com/noamkorngut/MRI_reconstruction.git   