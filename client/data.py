import os
import docker
from pathlib import Path
from urllib.parse import urlparse
import requests

from math import floor
import torch
import pandas as pd

TRAIN_DATA_FRAC = 0.8

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


# Helper function to download a file
def download_file(url, filename=None, filedir=None):
    if filename is None:
        a = urlparse(url)
        filename = os.path.basename(a.path)
    if filedir is not None:
        filename = os.path.join(filedir, filename)
    Path(filedir).mkdir(parents=True, exist_ok=True)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
    return filename


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # Check if the data is already downloaded
    if os.path.exists(os.path.join(out_dir, "train_dataset.pt")) and os.path.exists(
        os.path.join(out_dir, "test_dataset.pt")
    ):
        print(f"Files already downloaded to {out_dir}")
        return

    # Set URLs for the dataset( should set the url address of  data set here to download)
    # train_dataset_url = "www.example.com/train_dataset.pt"
    # test_dataset_url = "www.example.com/test_dataset.pt"

    # # Download the files to the data directory
    # train_dataset_file = download_file(train_dataset_url, "train_dataset.pt", out_dir)
    # test_dataset_file = download_file(test_dataset_url, "test_dataset.pt", out_dir)

    # print(f"Downloaded files {train_dataset_file} and {test_dataset_file} to {out_dir}")

    print(f"Downloaded files to {out_dir}")


def _get_data_path():
    """For test automation using docker-compose."""
    # Figure out FEDn client number from container name
    client = docker.from_env()
    container = client.containers.get(os.environ["HOSTNAME"])
    number = container.name[-1]

    # Return data path
    return f"/var/data/clients/{number}/UNSW_NB15.pt"


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get(
            "FEDN_DATA_PATH", abs_path + "/data/clients/1/UNSW_NB15.pt"
        )

    data = torch.load(data_path)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    return X, y


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    # calculate leftover data points
    leftover = n % parts

    lengths = [local_n + 1 if i < leftover else local_n for i in range(parts)]

    # result = torch.utils.data.random_split(dataset, lengths. torch.Generator().manual_seed(42))
    result = []

    start = 0
    for length in lengths:
        result.append(dataset[start : start + length])
        start += length

    return result


def split(out_dir="data", n_splits=2):

    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Load and convert to dict
    train_data = torch.load(f"{out_dir}/train_dataset.pt")
    test_data = torch.load(f"{out_dir}/test_dataset.pt")
    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(train_data.targets, n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(test_data.targets, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/UNSW_NB15.pt",
        )


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data()
        split()
