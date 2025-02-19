import os

file_path = "crop_health_model.h5"

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"File found: {file_path}")
