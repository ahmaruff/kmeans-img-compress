"""
Simple Image Compression Library using K-Means Clustering

A simple Python library for compressing images by reducing colors
using K-Means clustering algorithm.

Main Functions:
    compress: Main compression function
    get_compression_info: Get image information for compression planning
"""

from .core import (
    compress,
    get_compression_info,
    load_image,
    get_color_palette,
    quantize_img,
    save_quantized_img
)

__version__ = "0.1.0"
__author__ = "Ahmad Ma'ruf"
__email__ = "ahmadmaruf2701@gmail.com"

# Main functions for easy import
__all__ = [
    'compress',
    'get_compression_info',
    'load_image', 
    'get_color_palette',
    'quantize_img',
    'save_quantized_img'
]
