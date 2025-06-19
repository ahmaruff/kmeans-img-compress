# K-Means Image Compress

A simple Python library for image compression using K-Means clustering to reduce colors.

## Background

This library is derived from my undergraduate thesis research on image compression using K-Means clustering algorithm. The research aimed to evaluate the performance of K-Means algorithm in digital image compression by extracting dominant color palettes and applying color quantization for image reconstruction.

The study explored the impact of cluster numbers (K) on compression ratio and Peak Signal to Noise Ratio (PSNR) values. Through experimental methods, RGB 24-bit images with various resolutions (VGA, SVGA, HD, FHD) were compressed using different cluster numbers (8, 16, 32, 64, 96, 128).

**Key Research Results:**
- K-Means algorithm proved effective with average compression ratio of **67%**
- Average PSNR value of **81.43 dB** indicating good image quality retention
- **96 clusters emerged as the optimal value** providing the best balance between compression quality and computational efficiency

The default value of 96 clusters in this library is based on these empirical findings from my thesis research.

**Research Repository**: [Image Compression using K-Means](https://github.com/ahmaruff/image-compression-using-kmeans)  
**Paper**: [Ekstraksi Palet Warna untuk Kompresi Gambar Digital menggunakan Algoritma K-Means - Jurnal Informatika dan Rekayasa Perangkat Lunak](https://publikasiilmiah.unwahas.ac.id/JINRPL/article/view/12557)

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
```

## Quick Start

```python
import kmeans_img_compress as kic

# Basic compression with research-optimized default (96 clusters)
stats = kic.compress("input.jpg", "output.jpg")
print(f"Size reduced by {stats['size_reduction_percent']:.1f}%")

# Custom compression
stats = kic.compress("input.jpg", "output.jpg", n_colors=64, quality=80)
print(f"Compression ratio: {stats['compression_ratio']:.2f}")

# Get image info and recommendations
info = kic.get_compression_info("input.jpg")
print(f"Original size: {info['file_size_mb']:.2f} MB")
print(f"Recommended clusters: {info['recommended_clusters']}")
```

## Features

- Simple K-Means based color reduction
- Memory efficient processing for large images
- Multiple output formats (JPEG, PNG, WEBP)
- Detailed compression statistics
- Automatic parameter recommendations

## Requirements

- Python 3.7+
- NumPy
- Pillow
- scikit-learn

## License

[MIT License](./LICENSE)
