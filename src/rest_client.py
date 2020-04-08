import argparse
import base64
from typing import Dict
from pprint import pprint
import requests
import numpy as np
from tensorflow import keras


SERVER_URL = "http://localhost:8501/v1/models/classification:predict"


def encode_img(img_filename: str) -> str:
    with open(img_filename, "rb") as f:
        img_bytes = base64.b64encode(f.read())
    return img_bytes.decode("utf8")


def prepare_predict_request(img_filename: str) -> Dict:
    img_bytes = encode_img(img_filename)
    req = {
        "instances": [
            {"image_bytes": {"b64": img_bytes}},
        ]
    }
    return req


def send_predict_request(img_filename: str) -> Dict:
    predict_request = prepare_predict_request(img_filename)
    response = requests.post(SERVER_URL, json=predict_request)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="example REST client")
    parser.add_argument("--img", type=str, nargs="?",
                        help="path to input image")
    args = parser.parse_args()
    predictions = send_predict_request(args.img)["predictions"]
    pprint(keras.applications.imagenet_utils.decode_predictions(
        np.array(predictions)))


if __name__ == "__main__":
    main()
