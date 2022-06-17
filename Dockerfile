FROM usgsastro/miniconda3
WORKDIR /root

RUN apt-get --allow-releaseinfo-change update -y
RUN apt-get -y upgrade
RUN apt-get install -y libgomp1
RUN apt-get install -y libgl1-mesa-glx

ADD environment.yml /root
RUN conda install -c conda-forge mamba
RUN mamba env update -f environment.yml

ADD . /root
cmd pytest