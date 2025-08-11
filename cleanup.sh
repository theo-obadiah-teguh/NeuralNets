# Cleanup backups if they are not needed
rm -rf ./resnet/.ipynb_checkpoints
rm -rf ./tutorials/.ipynb_checkpoints

# Cleanup lightning checkpoints
rm -rf ./tutorials/lightning_logs