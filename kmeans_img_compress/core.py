"""
Image Compression Module using K-Means Clustering

This module provides simple functions to compress images by reducing colors
using K-Means clustering algorithm.

Functions:
    load_image: Load image from file path
    get_color_palette: Extract color palette using K-Means
    quantize_img: Apply color quantization to image
    save_quantized_img: Save quantized image to file
    compress: Main function to compress image
"""

import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def load_image(path: str) -> Image.Image:
    """
    Load image file from path.

    Args:
        path (str): Path to the image file

    Returns:
        Image.Image: PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image
        
    Example:
        >>> img = load_image("photo.jpg")
        >>> print(img.size)
        (1920, 1080)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        img = Image.open(path)
        img.verify()  # Check if valid image
        return Image.open(path)  # Reopen after verify
    except Exception as e:
        raise ValueError(f"Cannot load image from {path}") from e


def get_color_palette(
        img: Image.Image,
        n_colors: int = 96,
        max_pixels: int = 100000) -> Tuple[np.ndarray, KMeans]:
    """
    Extract color palette from image using K-Means clustering.
    
    Args:
        img (Image.Image): PIL Image object
        n_colors (int): Number of colors in palette (default: 96)
        max_pixels (int): Maximum pixels to use for K-Means (default: 100000)
        
    Returns:
        Tuple[np.ndarray, KMeans]: Color palette array and fitted KMeans model
        
    Raises:
        ValueError: If n_colors is not between 2-256
        
    Example:
        >>> palette, kmeans = get_color_palette(img, n_colors=64)
        >>> print(palette.shape)
        (64, 3)
    """
    if not (2 <= n_colors <= 256):
        raise ValueError("n_colors must be between 2 and 256")

    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize image for K-Means if too large (saves memory and processing time)
    width, height = img.size
    total_pixels = width * height

    if total_pixels > max_pixels:
        # Calculate resize ratio to stay under max_pixels
        ratio = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img_for_kmeans = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        img_for_kmeans = img

    # Convert image to numpy array and normalize to [0, 1]
    img_array = np.asarray(img_for_kmeans) / 255.0
    h, w, c = img_array.shape

    # Reshape to 2D array: (pixels, color_channels)
    pixels = img_array.reshape(h * w, c)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)

    # Get color palette (convert back to 0-255 range)
    palette = (kmeans.cluster_centers_ * 255).astype(np.uint8)

    return palette, kmeans


def quantize_img(img: Image.Image, kmeans: KMeans) -> np.ndarray:
    """
    Quantize image using pre-trained K-Means model.
    
    This function reduces the colors in an image by mapping each pixel
    to its nearest cluster center from the K-Means model.
    
    Args:
        img (Image.Image): PIL Image object to quantize
        kmeans (KMeans): Fitted K-Means model from get_color_palette()
        
    Returns:
        np.ndarray: Quantized image as numpy array (values in [0, 1] range)
        
    Example:
        >>> quantized = quantize_img(original_img, kmeans_model)
        >>> print(quantized.shape)
        (1080, 1920, 3)
    """
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array and normalize
    img_array = np.asarray(img) / 255.0
    h, w, c = img_array.shape

    # Reshape to 2D for prediction
    pixels = img_array.reshape(h * w, c)

    # Predict cluster labels for each pixel
    cluster_labels = kmeans.predict(pixels)

    # Replace pixels with their cluster centers
    quantized_pixels = kmeans.cluster_centers_[cluster_labels]

    # Reshape back to image dimensions
    return quantized_pixels.reshape(h, w, c)


def save_quantized_img(quantized_img: np.ndarray, filename: str, 
                      resize_factor: float = 1.0, quality: int = 85) -> None:
    """
    Save quantized image array to file.
    
    Args:
        quantized_img (np.ndarray): Quantized image array (values in [0, 1])
        filename (str): Output filename with extension
        resize_factor (float): Factor to resize final image (default: 1.0)
        quality (int): JPEG quality 1-100 (default: 85)
        
    Raises:
        ValueError: If resize_factor or quality are out of valid range
        
    Example:
        >>> save_quantized_img(quantized, "compressed.jpg", resize_factor=0.8, quality=80)
    """
    if not (0.1 <= resize_factor <= 1.0):
        raise ValueError("resize_factor must be between 0.1 and 1.0")
    if not (1 <= quality <= 100):
        raise ValueError("quality must be between 1 and 100")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to uint8 (0-255 range)
    img_uint8 = (quantized_img * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_uint8)

    # Resize if needed
    if resize_factor < 1.0:
        w, h = pil_img.size
        new_w = int(w * resize_factor)
        new_h = int(h * resize_factor)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Determine format from filename extension
    _, ext = os.path.splitext(filename.lower())

    # Save with appropriate settings
    if ext in ['.jpg', '.jpeg']:
        pil_img.save(filename, format='JPEG', optimize=True, quality=quality)
    elif ext == '.png':
        pil_img.save(filename, format='PNG', optimize=True)
    elif ext == '.webp':
        pil_img.save(filename, format='WEBP', optimize=True, quality=quality)
    else:
        # Default to JPEG
        pil_img.save(filename, format='JPEG', optimize=True, quality=quality)


def compress(source_img: str, output_img: str, n_colors: int = 96, 
            resize_factor: float = 1.0, quality: int = 85, 
            max_pixels_kmeans: int = 100000) -> dict:
    """
    Compress image using K-Means color quantization.
    
    This is the main function that combines all steps: loading the image,
    extracting color palette, quantizing colors, and saving the result.
    
    Args:
        source_img (str): Path to source image
        output_img (str): Path for compressed output image
        n_colors (int): Number of colors to use (default: 96)
        resize_factor (float): Factor to resize output (default: 1.0)
        quality (int): Output quality for JPEG (default: 85)
        max_pixels_kmeans (int): Max pixels for K-Means training (default: 100000)
        
    Returns:
        dict: Compression statistics including file sizes and compression ratio
        
    Raises:
        FileNotFoundError: If source image doesn't exist
        ValueError: If parameters are out of valid range
        
    Example:
        >>> stats = compress("photo.jpg", "compressed.jpg", n_colors=64, quality=80)
        >>> print(f"Compression ratio: {stats['compression_ratio']:.2f}")
        Compression ratio: 0.45
    """
    # Validate inputs
    if not os.path.exists(source_img):
        raise FileNotFoundError(f"Source image not found: {source_img}")

    # Get original file size
    original_size = os.path.getsize(source_img)

    # Load image
    img = load_image(source_img)

    # Extract color palette
    palette, kmeans = get_color_palette(img, n_colors, max_pixels_kmeans)

    # Quantize image
    quantized = quantize_img(img, kmeans)

    # Save compressed image
    save_quantized_img(quantized, output_img, resize_factor, quality)

    # Calculate compression statistics
    compressed_size = os.path.getsize(output_img)
    compression_ratio = compressed_size / original_size

    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - compression_ratio) * 100,
        'colors_used': n_colors,
    }


def get_compression_info(image_path: str) -> dict:
    """
    Get basic information about an image for compression planning.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Image information including dimensions, file size, etc.
        
    Example:
        >>> info = get_compression_info("photo.jpg")
        >>> print(f"Size: {info['dimensions']}, Colors: {info['estimated_colors']}")
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = load_image(image_path)
    file_size = os.path.getsize(image_path)
    
    # Convert to RGB to count unique colors
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Sample image to estimate unique colors (for large images)
    if img.size[0] * img.size[1] > 100000:
        # Resize for color estimation
        sample_img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img_array = np.asarray(sample_img)
    else:
        img_array = np.asarray(img)
    
    # Estimate unique colors
    pixels = img_array.reshape(-1, 3)
    unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*3)))))
    
    return {
        'dimensions': img.size,
        'mode': img.mode,
        'file_size_bytes': file_size,
        'file_size_mb': file_size / (1024 * 1024),
        'total_pixels': img.size[0] * img.size[1],
        'estimated_colors': min(unique_colors * 25, 16777216),
        'recommended_clusters': min(max(unique_colors // 10, 16), 128)
    }