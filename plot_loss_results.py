import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_loss_comparison(file_relu, file_elu):
    # Read the CSV files
    df_relu = pd.read_csv(file_relu)
    df_elu = pd.read_csv(file_elu)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Plot the loss values from both files
    ax.plot(df_relu.index, df_relu['loss'], label="Loss ReLU")
    ax.plot(df_elu.index, df_elu['loss'], label="Loss ELU")

    # Set labels for the x and y axes
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    # Add a legend
    ax.legend()

    # Save the plot to a PNG file
    fig.savefig("loss_comparison.png")

    # Optionally, close the plot
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare loss values from two CSV files.")
    parser.add_argument("file_relu", type=str, help="Path to the first (ReLU) CSV file")
    parser.add_argument("file_elu", type=str, help="Path to the second (ELU) CSV file")

    args = parser.parse_args()

    plot_loss_comparison(args.file_relu, args.file_elu)
