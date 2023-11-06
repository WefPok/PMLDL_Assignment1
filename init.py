import os

directories = [
    "data/external",
    "data/interim",
    "data/raw",
    "models",
    "notebooks",
    "references",
    "reports/figures",
    "src/data",
    "src/models",
    "src/visualization"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)