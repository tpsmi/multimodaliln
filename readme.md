# Illustrated London News Processing Project

This project contains a series of Jupyter notebooks for processing and analyzing images from the Illustrated London News collection. It covers the entire pipeline from data collection to multimodal search, enabling researchers to work with historical image datasets effectively.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Notebooks](#notebooks)
5. [Running the Code](#running-the-code)
6. [Troubleshooting](#troubleshooting)

## Project Overview

The Illustrated London News Processing Project aims to digitize, process, and analyze historical images from the Illustrated London News. This project demonstrates a complete workflow for working with large-scale historical image collections, including data acquisition, image processing, machine learning model training, and multimodal search capabilities.

## Requirements

- Python 3.11 or later
- Conda (Miniconda or Anaconda)
- Git (for cloning the repository)
- ImageMagick
- Tesseract OCR

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/illustrated-london-news-project.git
   cd illustrated-london-news-project
   ```

2. Create a Conda environment:
   ```
   conda create -n iln_project python=3.11
   ```

3. Activate the Conda environment:
   ```
   conda activate iln_project
   ```

4. Install the required packages using the requirements.txt file:
   ```
   pip install -r requirements.txt
   ```

5. Install ImageMagick:
   - For Mac: `brew install imagemagick`
   - For Windows: Download and install from [ImageMagick website](https://imagemagick.org/script/download.php)

6. Install Tesseract OCR:
   - For Mac: `brew install tesseract`
   - For Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Notebooks

1. `1-collectpagesfromIA.ipynb`: 
   - Purpose: Downloads zip files containing JPEG 2000 images from the Internet Archive.
   - Key Features:
     - Connects to the Internet Archive API
     - Finds and downloads zip files for a specified collection
     - Implements parallel downloading for efficiency
     - Includes error handling and download resumption capabilities

2. `2-jp2-to-jpeg-conversion.ipynb`:
   - Purpose: Converts JP2 (JPEG 2000) image files to standard JPEG format.
   - Key Features:
     - Uses ImageMagick for high-quality image conversion
     - Implements parallel processing for faster conversion
     - Includes progress tracking and error handling

3. `3-sampletrainandval.ipynb`:
   - Purpose: Randomly samples images and splits them into training and validation sets.
   - Key Features:
     - Implements stratified sampling to ensure balanced datasets
     - Customizable train/validation split ratio
     - Preserves directory structure in the sampled datasets

4. `4-finetuneyolov8.ipynb`:
   - Purpose: Trains and evaluates a YOLO (You Only Look Once) object detection model.
   - Key Features:
     - Uses the Ultralytics YOLOv8 implementation
     - Includes model training, evaluation, and export functionalities
     - Visualizes training results and model performance metrics

5. `5-image-extraction.ipynb`:
   - Purpose: Extracts images from a dataset using the trained YOLO model.
   - Key Features:
     - Applies the trained YOLO model to detect regions of interest
     - Extracts and saves detected regions as separate images
     - Generates metadata for extracted images

6. `6-ocrcaptiondetection.ipynb`:
   - Purpose: Performs text recognition on extracted images, focusing on captions.
   - Key Features:
     - Uses Tesseract OCR for text recognition
     - Implements image preprocessing to improve OCR accuracy
     - Handles text orientation detection and correction

7. `7-extraxt-image-embedding.ipynb`:
   - Purpose: Extracts image embeddings using the CLIP (Contrastive Language-Image Pre-training) model.
   - Key Features:
     - Utilizes the CLIP model to generate image embeddings
     - Implements batch processing for large datasets
     - Includes checkpointing for long-running processes

8. `8-multimodalsearch.ipynb`:
   - Purpose: Performs multimodal (text-to-image and image-to-image) similarity searches using CLIP embeddings.
   - Key Features:
     - Implements text-to-image search functionality
     - Implements image-to-image similarity search
     - Visualizes search results with similarity scores
   - Goal of Multimodal Retrieval:
     The primary goal is to enable flexible and intuitive searching of the image collection. Users can find relevant images using either text descriptions or example images. This multimodal approach allows for more nuanced and context-aware searches, bridging the gap between linguistic descriptions and visual content. It's particularly valuable for exploring historical images where traditional metadata might be limited or inconsistent.

## Running the Code

Follow these steps to run the notebooks in this project:

1. Ensure you're in the project directory:
   ```
   cd path/to/illustrated-london-news-project
   ```

2. Activate the Conda environment (if not already activated):
   ```
   conda activate iln_project
   ```

3. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

4. In the Jupyter interface, navigate to the project folder and open the notebooks in the following order:

   a. `1-collectpagesfromIA.ipynb`:
      - Set the `item_id` variable to the Internet Archive identifier for the collection you want to download.
      - Adjust the `download_path` variable to specify where you want to save the downloaded files.
      - Run all cells in the notebook to download the image files.

   b. `2-jp2-to-jpeg-conversion.ipynb`:
      - Set the `input_dir` variable to the folder containing your downloaded JP2 files.
      - Run all cells to convert the JP2 files to JPEG format.

   c. `3-sampletrainandval.ipynb`:
      - Set the `source_folder`, `destination_folder`, and `sample_size` variables as needed.
      - Run all cells to create your training and validation datasets.

   d. `4-finetuneyolov8.ipynb`:
      - Ensure your `illustrationmodel.yaml` file is correctly set up with paths to your training and validation data.
      - Adjust training parameters like `epochs` and `batch` size if needed.
      - Run all cells to train and evaluate the YOLO model.

   e. `5-image-extraction.ipynb`:
      - Set the `source` variable to the directory containing your original images.
      - Adjust the `output_folder` variable to specify where extracted images should be saved.
      - Run all cells to extract images using the trained YOLO model.

   f. `6-ocrcaptiondetection.ipynb`:
      - Ensure the `df` variable is loading the correct metadata CSV file.
      - Adjust the `output_folder` variable if necessary.
      - Run all cells to perform OCR on the extracted images.

   g. `7-extraxt-image-embedding.ipynb`:
      - Set the `source_folder` variable to the directory containing your extracted images.
      - Adjust the `checkpoint_path` and `final_path` variables as needed.
      - Run all cells to generate CLIP embeddings for your images.

   h. `8-multimodalsearch.ipynb`:
      - Ensure the `IMAGE_SOURCE_FOLDER` and `EMBEDDINGS_FILE` variables are set correctly.
      - Modify the `text_query` or `search_image_path` variables to perform different searches.
      - Run all cells to perform text-to-image and image-to-image searches.

5. After running each notebook, review the outputs and generated files to ensure everything worked as expected.

6. If you need to rerun a notebook, you can generally do so safely. However, be cautious with notebooks that involve downloading or processing large amounts of data, as you may want to skip those steps if they've already been completed.

### Important Notes:

- Each notebook is designed to be run sequentially, as later notebooks often depend on the outputs of earlier ones.
- Some processes, particularly in notebooks 4 and 7, can take a considerable amount of time to run. Plan accordingly and consider using a machine with a GPU for faster processing.
- Always check the input and output paths in each notebook to ensure they match your directory structure.
- If you encounter any errors, check the "Troubleshooting" section of this README or consult the comments within each notebook for guidance.
- It's recommended to save intermediate outputs (like converted images or trained models) in case you need to rerun parts of the pipeline without starting from scratch.

By following these steps, you should be able to process your image collection from raw downloads to searchable, analyzed data. Remember to adjust parameters and paths as necessary for your specific use case.

## Troubleshooting

- If you encounter issues with CUDA on Windows, make sure you have the correct CUDA toolkit installed for your GPU.
- On Mac with Apple Silicon, ensure you're using the correct version of TensorFlow (if needed) that supports ARM architecture.
- If `pytesseract` fails to find the Tesseract executable, set the path manually in the relevant notebook:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed
  ```
- If you're having trouble with the Conda environment, try creating a new one:
  ```
  conda create -n iln_project_new python=3.11
  conda activate iln_project_new
  pip install -r requirements.txt
  ```
- Ensure all file paths in the notebooks are correct and exist on your system.
- If you're working with a large dataset, make sure you have enough disk space available.

For any other issues, please open an issue on the GitHub repository.

