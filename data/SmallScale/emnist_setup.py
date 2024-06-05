# Download dataset and unzip it
# https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
# Move all files into the folder: move_to_your_foler/EMNIST/raw

ROOT = 'root/of/the/EMNIST/raw'
ROOT = '/local/scratch/hkim'

import os
import torchvision.datasets.utils as utils
import torchvision
import torchvision.transforms as transforms

def find_gz_files(folder_path):
    gz_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files

gz_files = find_gz_files(os.path.join(ROOT,'EMNIST','raw'))

if len(gz_files) > 0:
    for gz in gz_files:
        res = utils.extract_archive(gz, remove_finished=True)
        print(f"Extract {gz} -> {res}")

try:
    mnist = torchvision.datasets.EMNIST(
                root=ROOT,
                train=True,
                download=False, # RuntimeError: File not found or corrupted
                split="mnist",
                transform=transforms.ToTensor()
            )
    print("DONE!")
except:
    print("Issue to call EMNIST dataset from torchvision!")