import fiftyone as fo
import fiftyone.core.labels as fol
import os
import pandas as pd


default_fo_labels = ["plot_labels", "target_pred_labels", "black_corner_labels"]


def get_fo_dataset(dataset_name: str, black_corner_csv_fpath:str, plot_dirs:list, fo_labels:list=default_fo_labels):
    """
    Displays images stored in the heatmap, image, and overlayed folders and groups them based on the l[x] and p[x] values in the filename.
    
    Args:
    dataset_name: The name of the FiftyOne dataset
    black_corner_csv_fpath: The file path to the csv file containing the black corner labels.
    plot_dirs: A list of directories containing the source plot like images, gradcam heatmaps, and overlayed images.
    fo_labels: A dictionary containing the labels to be assigned to the samples. 
    """
    # Create a new FiftyOne dataset
    dataset = fo.Dataset(name=dataset_name)

    # Get the list of files in the heatmap directory
    file_names = os.listdir(plot_dirs[0])
    black_corner_df = pd.read_csv(black_corner_csv_fpath)

    # Read each file and add it to the dataset
    for file_name in file_names:
        # Assume filename format: ISIC_id_l[x]_p[x].ext

        # get the target pred label
        parts = file_name.split('_')
        lx = parts[-2][1:]  # l[x]
        px = parts[-1].split('.')[0][1:]  # p[x], remove extension
        target_pred_label_key = f"l{lx}_p{px}"

        # get the ISIC id
        image_id = "_".join(parts[:2])
        black_corner_label = black_corner_df[black_corner_df["image_id"] == image_id]["label"].values[0]

        # Load corresponding file paths
        for plot_dir in plot_dirs:
            # check if the plot label (category) is in the fo_labels
            plot_label = os.path.basename(plot_dir)
            
            # check if the plot exists
            plot_path = os.path.join(plot_dir, file_name)
            if not os.path.exists(plot_path):
                print(f"{plot_path} does not exist.")
                break
           
            # Create FiftyOne samples for each image type
            plot_sample = fo.Sample(filepath=plot_path)

            # add labels
            if "plot_labels" in fo_labels:
                plot_sample["plot_label"] = plot_label
            if "target_pred_labels" in fo_labels:
                plot_sample["target_pred_label"] = target_pred_label_key
            if "black_corner_labels" in fo_labels:
                plot_sample["black_corner"] = black_corner_label

            # Add samples to the dataset and the group
            dataset.add_sample(plot_sample)

    return dataset
