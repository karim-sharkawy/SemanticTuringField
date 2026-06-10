import os
import urllib.request
import zipfile
from functools import lru_cache

@lru_cache(maxsize=1)
def download_zip(): #Download zip file only once and return zipfile object
    url = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"
    zip_path = "data/glove.2024.wikigiga.50d.zip"
    
    if not os.path.exists(zip_path):
        print("Downloading GloVe from Stanford...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    else:
        print("Using cached zip file...")
    
    return zip_path

def extract_portion(lines_to_keep, name, zip_path):
    """Extract portion from already downloaded zip"""
    output_file = f"{name}.txt"
    
    print(f"Extracting {lines_to_keep:,} lines to {output_file}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Assuming the txt file has the same name pattern
        txt_file_name = f"{name.split('_')[0]}.txt"  # Adjust as needed
        txt_file_name = [f for f in zip_ref.namelist() if f.endswith('.txt')][0] # Or just find the first txt file
        
        with zip_ref.open(txt_file_name) as infile:
            with open(output_file, 'wb') as outfile:
                for i, line in enumerate(infile):
                    if i >= lines_to_keep:
                        break
                    outfile.write(line)
    
    print(f"Saved {lines_to_keep:,} lines to {output_file}")
    return output_file

if __name__ == "__main__":
    zip_path = download_zip()
    
    # Extract multiple portions from the same zip
    extract_portion(10_000, "glove.2024.wikigiga.50d_small", zip_path)
    #extract_portion(100_000, "glove.2024.wikigiga.50d_medium", zip_path)
    
    # delete file afterwards
    # os.remove(zip_path)