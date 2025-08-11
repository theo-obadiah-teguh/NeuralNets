# Build a Docker image based on the Dockerfile
docker build -t theothebear/nn-test .

# Run the Docker image as a container
# Map port 8888 in the Docker image to my local machine
# Sync the notebooks folder in the image, with the local notebooks folder
# Image name comes last
docker run -p 8888:8888 \
  -v "$(pwd)/resnet:/resnet" \
  -v "$(pwd)/tutorials:/tutorials" \
  theothebear/nn-test