import os
import requests
import zipfile
from tqdm import tqdm


def download_and_extract_dataset(root_dir, dataset_name, url):
    """
    Downloads and extracts a dataset from a given URL.

    Parameters:
        root_dir (str): Directory where the dataset will be stored.
        dataset_name (str): Name of the dataset for folder and file naming.
        url (str): URL of the dataset to download.
    """
    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Construct file paths
    zip_file_path = os.path.join(root_dir, f"{dataset_name}.zip")
    extract_dir = os.path.join(root_dir, dataset_name)

    # Skip download if zip file already exists
    if os.path.exists(extract_dir):
        print(f"Dataset '{dataset_name}' already exists at {extract_dir}. Skipping download and extraction.")
        return

    try:
        print(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        total_size = int(response.headers.get('content-length', 0))
        with open(zip_file_path, "wb") as file, tqdm(
                desc=f"Downloading {dataset_name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

        print("Download completed.")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        return

    # Extract the zip file
    try:
        print(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed.")
    except zipfile.BadZipFile as e:
        print(f"Error during extraction: {e}")
        if os.path.exists(extract_dir):
            os.rmdir(extract_dir)
        return

    # Optionally, clean up the zip file to save space
    os.remove(zip_file_path)
    print(f"Removed zip file: {zip_file_path}")


# Example usage
if __name__ == "__main__":
    datasets = {
        "COCO_Train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "COCO_Val2017": "http://images.cocodataset.org/zips/val2017.zip",
        # Add more datasets here
    }

    for name, url in datasets.items():
        download_and_extract_dataset("data", name, url)
