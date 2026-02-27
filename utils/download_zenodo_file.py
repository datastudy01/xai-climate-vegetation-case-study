import os
import requests
import zipfile


def download_zenodo_file(record_id: str, filename: str, extract_to: str = None):
    """
    Download a file from a Zenodo record draft into a folder and optionally unzip it.

    Args:
        record_id (str): Zenodo record ID.
        filename (str): Name of the zip file.
        extract_to (str, optional): Folder to download and extract into.
                                 If None, downloads into current folder.
    """
    # Determine folder and ensure it exists
    if extract_to is not None:
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, filename)
    else:
        zip_path = filename

    # Download if the zip doesn't already exist
    if os.path.exists(zip_path):
        print(f"{zip_path} already exists. Skipping download.")
    else:
        token = os.environ.get("ZENODO_TOKEN")
        if not token:
            raise ValueError("ZENODO_TOKEN environment variable not set.")

        url = f'https://zenodo.org/api/records/{record_id}/draft/files/{filename}/content'
        print(f"Starting download from {url}")

        with requests.get(url, params={"access_token": token}, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 // total
                            print(f"\rDownloading: {percent}%", end="", flush=True)
            if total:
                print("\rDownloading: 100%\nDownload complete.")
            else:
                print("\nDownload complete.")

        print(f"Downloaded {zip_path}")

    # Extract the zip if requested
    if extract_to is not None:
        if zipfile.is_zipfile(zip_path):
            print(f"Extracting {zip_path} to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extraction complete.")
        else:
            print(f"{zip_path} is not a valid zip file. Skipping extraction.")
