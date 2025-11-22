# upload_handler.py

import shutil
import os
import zipfile
from ipywidgets import FileUpload
from IPython.display import display


def upload_data(source_folder=None, upload_target_folder="new_uploads", use_gui=False):
    """
    Upload or copy images or folders to a target directory.

    Supports:
    - Uploading individual images
    - Uploading folders by zipping them first (GUI mode)
    - Copying entire folder structures when use_gui=False

    Parameters:
    - source_folder: str or None
        Path to an existing folder to copy from. Ignored in GUI mode.
    - upload_target_folder: str
        Folder where uploaded files will be saved.
    - use_gui: bool
        If True, shows a file upload widget for manual upload.
    """

    os.makedirs(upload_target_folder, exist_ok=True)

    # --------------------------------------------------------
    # GUI mode using Jupyter upload widget
    # --------------------------------------------------------
    if use_gui:
        upload_widget = FileUpload(
            accept="image/*,.zip",   # accept images or zipped folders
            multiple=True
        )

        display(upload_widget)

        def save_files(change):

            # Clear old uploads
            for item in os.listdir(upload_target_folder):
                path = os.path.join(upload_target_folder, item)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

            saved_count = 0

            for filename, file_info in upload_widget.value.items():
                file_path = os.path.join(upload_target_folder, filename)

                # Save raw file first
                with open(file_path, "wb") as f:
                    f.write(file_info["content"])

                # If the file is a zip, extract it
                if filename.lower().endswith(".zip"):
                    extract_folder = upload_target_folder
                    with zipfile.ZipFile(file_path, "r") as z:
                        z.extractall(extract_folder)
                    os.remove(file_path)  # remove zip after extracting
                    print(f"Extracted folder from {filename}")
                else:
                    print(f"Saved image: {filename}")

                saved_count += 1

            print(f"Upload complete. {saved_count} items saved to '{upload_target_folder}'.")

        upload_widget.observe(save_files, names="value")
        return upload_widget

    # --------------------------------------------------------
    # Non GUI mode, copying from an existing folder
    # --------------------------------------------------------
    else:
        if source_folder is None or not os.path.exists(source_folder):
            raise ValueError("source_folder must be provided and must exist when use_gui is False.")

        # Clear old uploads
        for item in os.listdir(upload_target_folder):
            path = os.path.join(upload_target_folder, item)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        # Copy everything recursively
        for item in os.listdir(source_folder):
            src = os.path.join(source_folder, item)
            dst = os.path.join(upload_target_folder, item)

            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

        print(f"Upload complete. Folder contents copied to '{upload_target_folder}'.")
