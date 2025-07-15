# Choose the Python base image. optimized for Python development
FROM python:latest

# Create a folder of notebooks
WORKDIR /

# Move the dependencies to the root folder
COPY requirements.txt /

# Run this before you copy notebooks, to use the layer caching system
# Note: Cached installation is not needed for Docker 
# (https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for)
RUN pip install --no-cache-dir -r requirements.txt jupyterlab

# Create a directory of all notebooks
RUN mkdir notebooks

# Move all notebooks from localhost to the image
COPY /notebooks/* /notebooks

# Expose the JupyterLab port
EXPOSE 8888

# Run jupyter lab, with the following flags
# --ip=0.0.0.0 - Listen on all network interfaces
# --no-browser - Don't try to launch a browser in container
# --allow-root - Necessary when running as root in container
# --NotebookApp.token='' - Disable authentication token
# --NotebookApp.password='' - Disable password authentication
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]