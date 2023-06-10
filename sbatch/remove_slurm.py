import os

listdir = os.listdir('.')
for file_ in listdir:
    if "slurm-" in file_:
        os.remove(file_)