# K-Means Image Compress

A simple Python library for image compression using K-Means clustering to reduce colors.

## Installation

### Install from GitHub

```bash
# Install directly from GitHub
pip install git+https://github.com/ahmaruff/kmeans-img-compress.git

# Or clone and install locally
git clone https://github.com/ahmaruff/kmeans-img-compress.git
cd kmeans-img-compress
pip install .

# For development (editable install)
pip install -e .

## Quick Start

```python
import kmeans_img_compress as kic

# Basic compression
stats = kic.compress("input.jpg", "output.jpg", n_colors=64)
print(f"Size reduced by {stats['size_reduction_percent']:.1f}%")

# Get image info first
info = kic.get_compression_info("input.jpg")
print(f"Recommended clusters: {info['recommended_clusters']}")
```

## Features

- Simple K-Means based color reduction
- Memory efficient processing
- Multiple output formats (JPEG, PNG, WEBP)
- Compression statistics
- Automatic parameter recommendations

## Requirements

- Python 3.7+
- NumPy
- Pillow
- scikit-learn

## License
[MIT License](./LICENSE)