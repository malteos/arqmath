# ARQMath with Transformers

Notebooks are available on Google Colab:
- [Training](https://colab.research.google.com/github/malteos/arqmath/blob/master/train.ipynb)
- [Inference](https://colab.research.google.com/github/malteos/arqmath/blob/master/inference.ipynb)


## Getting started

```bash
# Create conda env
conda create -n arqmath python=3.7
conda activate arqmath

# Clone repo
git clone https://github.com/malteos/arqmath.git
cd arqmath

# Install dependencies
pip install -r requirements.txt

# Download dataset
wget -O data/qa-pair.csv https://httpd.test.gipp.com/qa-pair.csv
```

## Training

```bash
jupyter notebook train.ipynb
```

## Inference

```bash
jupyter notebook inference.ipynb
```

## License

MIT
