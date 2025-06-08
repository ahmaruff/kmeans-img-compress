"""
Example usage of kmeans_img_compress library
"""

import kmeans_img_compress as kic

def main():
    """
    Example usage of kmeans_img_compress library
    """
    # Example image paths (change these to your actual images)
    input_path = "test.jpg"
    output_path = "compressed_image.jpg"

    try:
        # Get image info
        info = kic.get_compression_info(input_path)
        print(f"Original image: {info['dimensions']}")
        print(f"File size: {info['file_size_mb']:.2f} MB")
        print(f"Recommended clusters: {info['recommended_clusters']}")

        # Compress with different settings
        stats = kic.compress(
            input_path,
            output_path,
            n_colors=64,
            quality=80,
            resize_factor=0.8
        )

        print("\nCompression Results:")
        print(f"Size reduction: {stats['size_reduction_percent']:.1f}%")
        print(f"Colors used: {stats['colors_used']}")

    except FileNotFoundError:
        print("Please add a sample image file to test!")

if __name__ == "__main__":
    main()
