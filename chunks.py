import pandas as pd
import os

# split V-Dem-CD-v15.csv into 20 chunks
def split():
    input_file = "V-Dem-CD-v15.csv"
    output_dir = "chunks"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Get the total number of rows in the CSV file
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract 1 for the header
    chunk_size = total_rows // 20  # Calculate the size of each chunk

    reader = pd.read_csv(input_file, chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        chunk.to_csv(f'{output_dir}/chunk_{i}.csv', index=False)

split()