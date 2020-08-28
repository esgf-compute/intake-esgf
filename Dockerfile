FROM jupyter/minimal-notebook:latest

RUN conda install -c conda-forge -c jasonb857 intake-esgf ipywidgets

COPY examples/ .
