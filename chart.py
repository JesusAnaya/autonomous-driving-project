import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv files
data_1 = pd.read_csv('training_loss.csv')
data_2 = pd.read_csv('validation_loss.csv')

# Filter necessary columns
data_1 = data_1[['Step', 'Value']]
data_2 = data_2[['Step', 'Value']]

# Add source file identifier
data_1['Source'] = 'Training loss'
data_2['Source'] = 'Validation loss'

# Concatenate dataframes
data = pd.concat([data_1, data_2])

# Set plot style
sns.set(style="darkgrid")

# Create a line plot
plt.figure(figsize=(10,6))
sns.lineplot(x='Step', y='Value', hue='Source', data=data)

# Show the plot
plt.show()