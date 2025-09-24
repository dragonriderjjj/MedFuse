import matplotlib.pyplot as plt
import re
import numpy as np

def plot_comparative_training_results(log_file1, log_file2, label1, label2, output_prefix="comparison", max_epoch_plot=90):
    """
    Reads two log files, extracts training metrics (AUROC, AUPRC, Accuracy),
    and generates comparative plots.  Saves the plots to files.
    Only plots up to max_epoch_plot.

    Args:
        log_file1 (str): Path to the first log file.
        log_file2 (str): Path to the second log file.
        label1 (str): Label for the first log file in the plots.
        label2 (str): Label for the second log file in the plots.
        output_prefix (str): Prefix for the output file names.
        max_epoch_plot (int): Maximum epoch to plot up to.
    """

    def parse_log_file(log_file):
        epochs = []
        train_auroc = []
        valid_auroc = []
        train_auprc = []
        valid_auprc = []
        train_acc = []
        valid_acc = []

        with open(log_file, "r") as f:
            epoch = None  # Keep track of the current epoch
            train_metrics = None # Keep track of training metrics
            for line in f:
                # Extract epoch
                epoch_match = re.search(r"Epoch \[(.*?)/", line)
                if epoch_match:
                    try:
                        epoch = int(epoch_match.group(1))
                        if epoch > max_epoch_plot:
                            continue  # Skip epochs beyond the plotting limit
                        
                        # Extract training metrics
                        train_metrics_match = re.search(r"\(train\) loss = .*?, accuracy = (.*?), AUROC = (.*?), AUPRC = (.*)", line)
                        if train_metrics_match:
                            try:
                                train_acc_val = float(train_metrics_match.group(1))
                                train_auroc_val = float(train_metrics_match.group(2))
                                train_auprc_val = float(train_metrics_match.group(3))
                                train_metrics = (train_acc_val, train_auroc_val, train_auprc_val)

                            except ValueError:
                                print(f"Warning: Could not convert train metrics to float from line: {line}")
                                train_metrics = None # Reset train_metrics if extraction fails

                        else:
                            train_metrics = None # Reset train_metrics if no match
                            print(f"Warning: Could not extract train metrics from line: {line}")
                    except ValueError:
                        print(f"Warning: Could not convert epoch to int from line: {line}")
                        epoch = None
                # Extract validation metrics
                
                try:
                    valid_metrics_match = re.search(r"^\s*\| \(valid\) loss = .*?, accuracy = (.*?), AUROC = (.*?), AUPRC = (.*)", line)
                    if valid_metrics_match:
                        try:
                            valid_acc_val = float(valid_metrics_match.group(1))
                            valid_auroc_val = float(valid_metrics_match.group(2))
                            valid_auprc_val = float(valid_metrics_match.group(3))

                            # Check if we have valid epoch and train metrics
                            if epoch is not None and train_metrics is not None:
                                epochs.append(epoch)  # Append epoch only if within limit and metrics are valid
                                train_acc.append(train_metrics[0])
                                train_auroc.append(train_metrics[1])
                                train_auprc.append(train_metrics[2])
                                valid_acc.append(valid_acc_val)
                                valid_auroc.append(valid_auroc_val)
                                valid_auprc.append(valid_auprc_val)
                                print(f"Epoch: {epoch}, Train AUROC: {train_metrics[1]}, Valid AUROC: {valid_auroc_val}")  # Debugging print
                                train_metrics = None # Reset train_metrics after appending

                        except ValueError:
                            print(f"Warning: Could not convert valid metrics to float from line: {line}")
                except:
                    pass

        print(f"Extracted {len(epochs)} epochs, {len(train_auroc)} train_auroc, {len(valid_auroc)} valid_auroc values from {log_file}")
        return epochs, train_auroc, valid_auroc, train_auprc, valid_auprc, train_acc, valid_acc

    # Parse the log files
    epochs1, train_auroc1, valid_auroc1, train_auprc1, valid_auprc1, train_acc1, valid_acc1 = parse_log_file(log_file1)
    epochs2, train_auroc2, valid_auroc2, train_auprc2, valid_auprc2, train_acc2, valid_acc2 = parse_log_file(log_file2)

    # Find the maximum number of epochs, but limit to max_epoch_plot
    max_epochs = min(max(len(epochs1), len(epochs2)), max_epoch_plot)

    # Interpolate the data
    def interpolate_data(epochs, data, max_epochs):
        if not epochs or not data:
            return [None] * max_epochs  # Return a list of None values if data is empty
        
        # Filter epochs and data to be within the plotting limit
        epochs = [e for e in epochs if e <= max_epochs]
        data = data[:len(epochs)]

        if len(epochs) == max_epochs:
            return data  # No interpolation needed

        if not epochs:
            return [None] * max_epochs

        # Convert to numpy arrays
        epochs = np.array(epochs)
        data = np.array(data)

        # Create new epochs
        new_epochs = np.linspace(epochs.min(), epochs.max(), max_epochs)

        # Interpolate the data
        interpolated_data = np.interp(new_epochs, epochs, data)

        return interpolated_data.tolist()

    # Interpolate the data

    # Create a common set of epochs for plotting
    common_epochs = list(range(1, max_epochs + 1))

    # Create plots
    plt.figure(figsize=(18, 6))

    # Plot AUROC
    plt.subplot(1, 3, 3)
    if common_epochs:
        # plt.plot(common_epochs, train_auroc1, label=f"{label1} (Train)", linestyle='-', marker='o', markersize=3)
        plt.plot(common_epochs, valid_auroc1, label=f"{label1} (Valid)", linestyle='-', marker='o', markersize=3)
        # plt.plot(epochs2, train_auroc2, label=f"{label2} (Train)", linestyle='--', marker='s', markersize=3)
        plt.plot(epochs2, valid_auroc2, label=f"{label2} (Valid)", linestyle='--', marker='s', markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("AUROC")
        plt.legend()
        plt.title("Comparative AUROC vs. Epoch")
        plt.grid(True)

    # Plot AUPRC
    plt.subplot(1, 3, 2)
    if common_epochs:
        # plt.plot(common_epochs, train_auprc1, label=f"{label1} (Train)", linestyle='-', marker='o', markersize=3)
        plt.plot(common_epochs, valid_auprc1, label=f"{label1} (Valid)", linestyle='-', marker='o', markersize=3)
        # plt.plot(epochs2, train_auprc2, label=f"{label2} (Train)", linestyle='--', marker='s', markersize=3)
        plt.plot(epochs2, valid_auprc2, label=f"{label2} (Valid)", linestyle='--', marker='s', markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("AUPRC")
        plt.legend()
        plt.title("Comparative AUPRC vs. Epoch")
        plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 1)
    if common_epochs:
        # plt.plot(common_epochs, train_acc1, label=f"{label1} (Train)", linestyle='-', marker='o', markersize=3)
        plt.plot(common_epochs, valid_acc1, label=f"{label1} (Valid)", linestyle='-', marker='o', markersize=3)
        # plt.plot(epochs2, train_acc2, label=f"{label2} (Train)", linestyle='--', marker='s', markersize=3)
        plt.plot(epochs2, valid_acc2, label=f"{label2} (Valid)", linestyle='--', marker='s', markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Comparative Accuracy vs. Epoch")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison_plots.png")
    plt.close()

# Example usage:
log_file1 = "/home/ehr/SCANE_backup/exp_output/overall_result/20250527_P12_TranSCANE_no_rope/training_result/debug.log"  # Replace with your actual file paths
log_file2 = "/home/ehr/SCANE/debug.log"  # Replace with descriptive labels
label1 = "SCANE"  # Replace with descriptive labels
label2 = "RSCANE"  # Replace with descriptive labels
output_prefix = "my_model" # Replace with your desired output prefix

plot_comparative_training_results(log_file1, log_file2, label1, label2, output_prefix, max_epoch_plot=250)