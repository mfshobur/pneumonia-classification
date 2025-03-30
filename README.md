# Pneumonia Classification

This project is to classify an X-ray images into 2 categories (Pneumonia and Normal).

Model used is pre-trained resnet18 model and then fine-tuned using Pneumonia Dataset

The dataset source is from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Installation

Use the package manager [pip](https://pypi.org/project/pip/) to install required packages.

```bash
pip3 install -r requirements.txt
```

## Usage

Run the Flask webApp using this command:

```bash
cd backend
python3 main.py
```

Open the web if already running at [http://localhost:5000](http://localhost:5000)

## Screenshots

![alt text](image.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)