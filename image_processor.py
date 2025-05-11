import cv2
import os
import time
import logging
from multiprocessing import Pool, cpu_count
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Constants
INPUT_DIR = "images"
OUTPUT_DIR = "output"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# === FILTER PROCESSING ===

def apply_filters(image_path: str) -> None:
    """
    Applies a series of image filters to the input image and saves the results in separate folders.
    """
    try:
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"⚠️ Could not read image: {image_path}")
            return

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        # Filter 1: Gaussian Blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        save_image(blurred, "gaussian_blur", name, ext)

        # Filter 2: Brightness Adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
        save_image(bright, "brightness_adjusted", name, ext)

        logging.info(f"✅ Processed: {filename}")

    except Exception as e:
        logging.error(f"❌ Error processing {image_path}: {e}")


def save_image(image, filter_name: str, base_name: str, ext: str) -> None:
    """
    Saves a processed image into a subfolder under the main output directory.

    Args:
        image: Image to save.
        filter_name: Name of the filter (used as subfolder name).
        base_name: Base filename (without extension).
        ext: File extension.
    """
    folder = os.path.join(OUTPUT_DIR, filter_name)
    os.makedirs(folder, exist_ok=True)

    output_path = os.path.join(folder, f"{base_name}{ext}")
    cv2.imwrite(output_path, image)


# === UTILS ===

def get_image_paths(directory: str) -> List[str]:
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(SUPPORTED_EXTENSIONS)
    ]


# === PROCESSING MODES ===

def process_images_sequentially(image_paths: List[str]) -> float:
    logging.info("\n⏳ Starting sequential processing...")
    start_time = time.perf_counter()

    for path in image_paths:
        apply_filters(path)

    duration = time.perf_counter() - start_time
    logging.info(f"🕒 Sequential processing completed in {duration:.2f} seconds.\n")

    return duration


def process_images_in_parallel(image_paths: List[str]) -> float:
    logging.info("\n⚙️  Starting parallel processing...")
    start_time = time.perf_counter()

    with Pool(cpu_count()) as pool:
        pool.map(apply_filters, image_paths)

    duration = time.perf_counter() - start_time
    logging.info(f"🚀 Parallel processing completed in {duration:.2f} seconds.\n")

    return duration


# === MAIN ===

def main():
    print("📸 Parallel Image Filter Processor")
    print("====================================\n")

    if not os.path.exists(INPUT_DIR):
        logging.error(f"📁 Input directory '{INPUT_DIR}' not found. Please create it and add images.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = get_image_paths(INPUT_DIR)

    if not image_paths:
        logging.warning("📭 No images found in the input directory.")
        return

    logging.info(f"📂 Found {len(image_paths)} images in '{INPUT_DIR}'.")

    # Sequential
    sequential_time = process_images_sequentially(image_paths)

    # Parallel
    parallel_time = process_images_in_parallel(image_paths)

    # Performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")

    print("📊 Performance Summary")
    print("------------------------")
    print(f"🧵 Sequential time: {sequential_time:.2f} seconds")
    print(f"🧩 Parallel time:   {parallel_time:.2f} seconds")
    print(f"⚡ Speedup:          {speedup:.2f}x\n")
    print("✅ All filters applied and results saved by filter type in the 'output/' folder.")


if __name__ == "__main__":
    main()
