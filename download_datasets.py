import os
import requests
import zipfile
from tqdm import tqdm

def download_dataset(root_dir, dataset_name, url):
    os.makedirs(root_dir, exist_ok=True)
    zip_file_path = os.path.join(root_dir, f"{dataset_name}.zip")
    extract_dir = os.path.join(root_dir, dataset_name)

    if os.path.exists(extract_dir):
        print(f"Dataset '{dataset_name}' already exists at {extract_dir}.")
        return

    try:
        print(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_file_path, "wb") as file, tqdm(
            desc=f"Downloading {dataset_name}",
            total=int(response.headers.get('content-length', 0)),
            unit='iB', unit_scale=True, unit_divisor=1024
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(file.write(data))

        print("Download completed. Extracting...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print("Extraction completed.")

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)

# Example usage
if __name__ == "__main__":
    datasets = {
        "COCO_Train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "COCO_Val2017": "http://images.cocodataset.org/zips/val2017.zip"
    }
    for name, url in datasets.items():
        download_dataset("data", name, url)
