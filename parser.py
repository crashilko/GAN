
def parse(cfg_path):
    with open(cfg_path, 'r') as f:
        file = f.readlines()
        print(file)
    pass


import tensorflow_datasets as tfds
from parser import parse

