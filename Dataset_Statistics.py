import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Obtains statistics about properties of any provided datasets and produces corresponding charts
def count_images_in_species(root_dir):
    species_counts = {}
    total_count = 0

    # Walk through the directory structure
    for species in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species)

        if os.path.isdir(species_path):
            image_count = 0

            # Go through the individual folders inside each species folder
            for individual in os.listdir(species_path):
                individual_path = os.path.join(species_path, individual)

                if os.path.isdir(individual_path):
                    # Count all files in the individual folder
                    image_count += len([f for f in os.listdir(individual_path) if os.path.isfile(os.path.join(individual_path, f))])
                    image_count -= 1

            species_counts[species] = image_count
            total_count += image_count
    print("Total images (from ALL species): ", total_count)
    return species_counts

def plot_image_counts_species(species_counts):
    species = list(species_counts.keys())
    counts = list(species_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(species, counts, color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Species')
    plt.xticks(rotation=90)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

    plt.tight_layout()
    plt.show()

def count_images_in_individuals(root_dir, species):
    individual_counts = {}
    if species == "All species":
        # Walk through the directory structure and count individuals for all species present
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)

            if os.path.isdir(species_path):
                # Go through the individual folders inside each species folder
                for individual in os.listdir(species_path):
                    individual_path = os.path.join(species_path, individual)

                    if os.path.isdir(individual_path):
                        # Count all files in the individual folder
                        image_count = len([f for f in os.listdir(individual_path) if os.path.isfile(os.path.join(individual_path, f))])

                        # Use the individual folder name as a key, appending the species name to avoid name conflicts
                        individual_name = f"{individual}"
                        individual_counts[individual_name] = image_count
        return individual_counts

    # If a specific species was specified in the parameters
    species_path = os.path.join(root_dir, species)
    for individual in os.listdir(species_path):
        individual_path = os.path.join(species_path, individual)

        if os.path.isdir(individual_path):
            # Count all files in the individual folder
            image_count = len([f for f in os.listdir(individual_path) if os.path.isfile(os.path.join(individual_path, f))])

            # Use the individual folder name as a key, appending the species name to avoid name conflicts
            individual_name = f"{individual}"
            individual_counts[individual_name] = image_count

    return individual_counts


def plot_image_counts_individuals(species="All species"):
    individual_counts = count_images_in_individuals(root_directory, species)
    sorted_individuals = sorted(individual_counts.items(), key=lambda x: x[0])
    individuals, counts = zip(*sorted_individuals)  # Unpack sorted data

    plt.figure(figsize=(12, 8))
    bars = plt.barh(individuals, counts, color='skyblue', height=0.8)
    # Set the y-axis limits to reduce whitespace
    plt.ylim(-0.5, len(individuals) - 0.5)

    # Remove the y-axis labels (individual names)
    plt.yticks([])

    plt.ylabel('Individual')
    plt.xlabel('Number of Images')
    plt.title('Number of Images per Individual ('+species+')')

    # Annotate each bar with its value
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{int(bar.get_width())}", va='center', ha='left')

    plt.tight_layout()
    plt.show()

def calculate_average_images_per_individual(root_dir):
    species_data = defaultdict(lambda: {'total_images': 0, 'num_individuals': 0})

    # Walk through the directory structure
    for species in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species)

        if os.path.isdir(species_path):
            # Go through the individual folders inside each species folder
            for individual in os.listdir(species_path):
                individual_path = os.path.join(species_path, individual)

                if os.path.isdir(individual_path):
                    # Count all files in the individual folder
                    image_count = len([f for f in os.listdir(individual_path) if os.path.isfile(os.path.join(individual_path, f))])
                    image_count -= 1  # Get rid of image_count including 1 for the current folder it is in
                    # Update the total images and number of individuals for the species
                    species_data[species]['total_images'] += image_count
                    species_data[species]['num_individuals'] += 1

    # Calculate the average number of images per individual for each species
    species_averages = {}
    for species, data in species_data.items():
        if data['num_individuals'] > 0:
            species_averages[species] = data['total_images'] / data['num_individuals']
        else:
            species_averages[species] = 0

    return species_averages

def plot_species_averages(species_averages):
    species = list(species_averages.keys())
    averages = list(species_averages.values())

    plt.figure(figsize=(12, 8))
    bars = plt.bar(species, averages, color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Average Number of Images per Individual')
    plt.title('Average Number of Images per Individual per Species')
    plt.xticks(rotation=90)

    # Annotate each bar with its value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", va='bottom', ha='center')

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    # Set the root directory path here
    root_directory = 'Datasets/AFD/'

    # Number of images per species ------------------------------------
    # species_counts = count_images_in_species(root_directory)
    # plot_image_counts_species(species_counts)

    # Number of images per individual --------------------------------
    plot_image_counts_individuals("Rhinopithecus roxellanae")

    # Average number of images per individual, grouped with the x-axis being each species
    # species_averages = calculate_average_images_per_individual(root_directory)
    # plot_species_averages(species_averages)