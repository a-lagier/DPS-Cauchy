import os
import yaml
import argparse


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_parser() -> dict:
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--cfg", type=str)
    args = parser_.parse_args()

    return load_yaml(args.cfg)

def get_sample_size(cfg: dict) -> tuple:
    batch_size = cfg["batch_size"]
    channels = cfg["channels"]
    sampling_size = cfg["sampling_size"]

    return (batch_size, channels, sampling_size, sampling_size)