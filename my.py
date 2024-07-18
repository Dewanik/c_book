import argparse
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import pandas as pd

# Function to load page titles from a CSV or TXT file
def load_titles(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        titles = df.iloc[:, 0].tolist()
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            titles = file.readlines()
    else:
        titles = []
    return [title.strip() for title in titles]

# Function to generate coloring book images
def generate_coloring_book(titles):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16
    ).to('cuda')
    pipeline.load_lora_weights('artificialguybr/ColoringBookRedmond-V2', weight_name='ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors')
    for title in titles:
        image = pipeline(f'{title}, Coloring Book, ColoringBookAF').images[0]
        image.save(f"{title.replace(' ', '_')}_coloring_book.png")

# Command-line interface for input
def get_input():
    parser = argparse.ArgumentParser(description='Generate Coloring Book Pages')
    parser.add_argument('--single', type=str, help='Enter a single page title')
    parser.add_argument('--file', type=str, help='Path to a CSV or TXT file containing page titles')
    args = parser.parse_args()

    if args.single:
        generate_coloring_book([args.single])
    elif args.file:
        titles = load_titles(args.file)
        generate_coloring_book(titles)
    else:
        print("Please provide a single title or a file path.")

# Run the command-line interface
get_input()
