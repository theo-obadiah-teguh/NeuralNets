# Build a Docker image based on the Dockerfile
docker build -t theothebear/nn-test .

# Run the Docker image as a container
# Map port 8888 in the Docker image to my local machine
docker run -p 8888:8888 theothebear/nn-test