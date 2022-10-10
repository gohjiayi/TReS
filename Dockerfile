# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.07-py3

# Install dependencies (pip or conda)
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# RUN conda update -n base conda
# COPY environment.yml environment.yml
# RUN conda env create -f environment.yml
# RUN conda activate tres