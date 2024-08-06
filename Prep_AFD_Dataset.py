import os
import shutil

# Renames the appropriate folders to organize the classes for the AFD dataset
def rename_folders(base_path):
    # Loop through each species directory
    for species_dir in os.listdir(base_path):
        species_path = os.path.join(base_path, species_dir)

        if os.path.isdir(species_path):
            # Get the species name
            species_name = species_dir

            # Loop through each numbered folder inside the species directory
            for folder_name in os.listdir(species_path):
                folder_path = os.path.join(species_path, folder_name)

                if os.path.isdir(folder_path):
                    # Construct the new folder name
                    new_folder_name = f"{species_name} - {folder_name}"
                    new_folder_path = os.path.join(species_path, new_folder_name)

                    # Rename the folder
                    os.rename(folder_path, new_folder_path)
            print("Renamed all directories")

# Moves all images so that they are in the parent directory instead of a "ver" subfolder
def move_images(base_path):
    # Loop through each species directory
    for species_dir in os.listdir(base_path):
        species_path = os.path.join(base_path, species_dir)

        if os.path.isdir(species_path):
            # Loop through each numbered folder inside the species directory
            for folder_name in os.listdir(species_path):
                folder_path = os.path.join(species_path, folder_name)

                if os.path.isdir(folder_path):
                    ver_folder_path = os.path.join(folder_path, 'test')

                    if os.path.isdir(ver_folder_path):
                        # Move all files from 'ver' folder to the parent folder
                        for file_name in os.listdir(ver_folder_path):
                            file_path = os.path.join(ver_folder_path, file_name)
                            if os.path.isfile(file_path):
                                # Construct the new path in the parent folder
                                new_file_path = os.path.join(folder_path, file_name)
                                shutil.move(file_path, new_file_path)
                                print(f"Moved {file_path} to {new_file_path}")

                        # Remove the empty 'ver' folder
                        os.rmdir(ver_folder_path)
                        print(f"Removed empty 'ver' folder: {ver_folder_path}")

# rename_folders("AFD/")
# move_images("AFD/")

