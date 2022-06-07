______________________________________________________________________
<div align="center">

# 👨‍🎨 Edge-aware Face Autoencoder

<p align="center">
  <a href="https://github.com/wiktorlazarski">👨‍🎓 Wiktor</a>
  <a href="https://github.com/AnetaJas">👩‍🎓 Aneta</a>
</p>

______________________________________________________________________

[![ci-testing](https://github.com/wiktorlazarski/face-vae/actions/workflows/ci-testing.yml/badge.svg?branch=master&event=push)](https://github.com/wiktorlazarski/edge-aware-face-autoencoder/actions/workflows/ci-testing.yml)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b3afUvKmuWQblkxIpGxUAOt0pZk3DBa3?usp=sharing)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 💎 Installation with `pip`

Installation is as simple as running:

```bash
pip install git+https://github.com/wiktorlazarski/edge-aware-face-autoencoder.git
```

## 🧐 Qualitative results

#### Original images
![alt text](https://github.com/wiktorlazarski/edge-aware-face-autoencoder/blob/master/doc/images/original_imgs.png)

---

#### Baseline model without edge-awareness trained on images with 256px resolution, VAE latent dimension equals to 512 and reconstruction loss weight equals to 100 000. Edges' weight was set to 1.
![alt text](https://github.com/wiktorlazarski/edge-aware-face-autoencoder/blob/master/doc/images/baseline-256px-512ld-100k.png)

---

#### Edge-aware model with edges' weight set to 3. All other parameters were the same as in the baseline model.
![alt text](https://github.com/wiktorlazarski/edge-aware-face-autoencoder/blob/master/doc/images/edgeaware-edge_weight3.png)

---

#### Edge-aware model with edges' weight set to 10. All other parameters were the same as in the baseline model.
![alt text](https://github.com/wiktorlazarski/edge-aware-face-autoencoder/blob/master/doc/images/edgeaware_edge_weight10.png)

---

## ⚙️ Setup for development with `pip`

```bash
# Clone repo
git clone https://github.com/wiktorlazarski/edge-aware-face-autoencoder.git

# Go to repo directory
cd edge-aware-face-autoencoder

# (Optional) Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install project in editable mode
pip install -e .[dev]

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

## 🐍 Setup for development with `conda`

```bash
# Clone repo
git clone https://github.com/wiktorlazarski/edge-aware-face-autoencoder.git

# Go to repo directory
cd edge-aware-face-autoencoder

# Create and activate conda environment
conda env create -f ./conda_env.yml
conda activate face_autoencoder

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

<div align="center">

### 🤗 Enjoy the model !

</div>
