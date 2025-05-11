# 📸 Parallel Image Filter Processor

This project demonstrates how to apply image filters in both **sequential** and **parallel** modes using Python's `multiprocessing` module and OpenCV. It helps visualize performance benefits and organizes results by filter type.

## 🚀 Features

- ✅ Gaussian Blur filter  
- ✅ Brightness Adjustment  
- ✅ Sequential and parallel processing modes  
- ✅ Organized output folders by filter type  
- ✅ Terminal-friendly logs for improved UX

## 📁 Project Structure

```
.
├── images/             # Input images
├── output/             # Output folder with filter subdirectories
│   ├── gaussian_blur/
│   └── brightness_adjusted/
├── image_processor.py  # Main script
└── image_processor.md           # You're reading it
```

## 🧪 How to Use

1. Place your `.jpg`, `.jpeg`, or `.png` images into the `images/` directory.
2. Run the script:
   ```bash
   python image_processor.py
   ```
3. Check the `output/` folder for processed results.

## 📦 Requirements

- Python 3.6+
- OpenCV

Install with:
```bash
pip install opencv-python
```

## 📌 Example Output Log

```
📸 Parallel Image Filter Processor
====================================
2025-05-11 19:09:33,955 | 📂 Found 5 images in 'images'.
2025-05-11 19:09:33,955 | 
⏳ Starting sequential processing...
2025-05-11 19:09:33,967 | ✅ Processed: image1.jpg
2025-05-11 19:09:33,969 | ✅ Processed: image2.jpg
2025-05-11 19:09:33,970 | ✅ Processed: image3.jpg
2025-05-11 19:09:33,973 | ✅ Processed: image4.png
2025-05-11 19:09:33,975 | ✅ Processed: image5.jpg
2025-05-11 19:09:33,975 | 🕒 Sequential processing completed in 0.02 seconds.
2025-05-11 19:09:33,975 | 
⚙️  Starting parallel processing...
2025-05-11 19:09:34,458 | ✅ Processed: image1.jpg
2025-05-11 19:09:34,461 | ✅ Processed: image3.jpg
2025-05-11 19:09:34,466 | ✅ Processed: image2.jpg
2025-05-11 19:09:34,467 | ✅ Processed: image4.png
2025-05-11 19:09:34,469 | ✅ Processed: image5.jpg
2025-05-11 19:09:34,478 | 🚀 Parallel processing completed in 0.50 seconds.
```

## 📊 Performance Summary

------------------------
- 🧵 Sequential time: 0.02 seconds
- 🧩 Parallel time:   0.50 seconds
- ⚡ Speedup:          0.04x

✅ All filters applied and results saved by filter type in the 'output/' folder.

## ✅ Output Sample (File Tree)

```
output/
├── gaussian_blur/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── brightness_adjusted/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

## 📝 Implementation Details

The script utilizes Python's `multiprocessing` module to leverage multi-core processing for image filter applications. The implementation compares the performance difference between:

1. **Sequential processing**: Images are processed one after another
2. **Parallel processing**: Multiple images are processed simultaneously using available CPU cores

Each filter is applied independently and results are organized in separate directories for easy comparison.

## 🛠️ Extending the Project

To add new filters:
1. Add a new filter function in `image_processor.py`
2. Create a corresponding output directory
3. Update the filter application code to include your new filter

## 📊 Performance Considerations

Note that for small images or simple filters, the overhead of creating processes might outweigh the benefits of parallel processing. This is why you might see better performance with sequential processing in some cases.

For best results with parallel processing:
[- Use larger images]()
- Apply more computationally intensive filters
- Process larger batches of images

## 📜 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.