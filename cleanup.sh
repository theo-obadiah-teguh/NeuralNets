# Cleanup backups if they are not needed
rm -rf ./notebooks/.ipynb_checkpoints
rm -rf ./notebooks/ResNet/.ipynb_checkpoints
rm -rf ./notebooks/Tutorials/.ipynb_checkpoints

# Cleanup lightning checkpoints
rm -rf ./notebooks/lightning_logs