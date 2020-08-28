FROM jupyter/minimal-notebook:latest

RUN conda create -n intake-esgf -c conda-forge -c jasonb857 intake-esgf ipywidgets
