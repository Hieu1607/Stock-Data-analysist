import numpy as np
import matplotlib.pyplot as plt
import random

def generate_random_color():
    return (random.random(), random.random(), random.random())

def plot_histogram(data_list, specific_data=None, bin_width=1000 , criteria=None,number_of_bins=30):
    """
    Visualize a histogram of total profits, group similar values, and compare with in-sample profit.
    
    Parameters:
    - data_list: List or array of total profits (e.g., 100 values).
    - in_sample_profit: The in-sample total profit to compare (optional).
    - bin_width: Width of each bin for grouping similar profits (default: 0.01).
    """
    if number_of_bins is not None:
        number_of_bins = number_of_bins
    if criteria is None:
        criteria = 'total_profit'
    else:
        criteria = criteria

    # Determine the bins for grouping
    min_val = 0
    max_val = 0
    min_data = min(data_list)
    max_data = max(data_list)
    min_val = min_data - bin_width
    max_val = max_data + bin_width
    bins = np.arange(min_val,max_val, (max_val - min_val) / number_of_bins)  # Create bins from min to max with specified width
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(data_list, bins=bins, density=True, alpha=0.7, color='gray', edgecolor='black')
    
    # Customize the plot
    plt.xlabel(f'{criteria} (Ω)')
    plt.ylabel('Probability')
    plt.title(f'Histogram of {criteria}')
    
    # If data is provided, mark it and calculate p-value
    if specific_data is not None:
        for key, value in specific_data.items():
            # Mark the data on the histogram
            data = float(value)  # Ensure it's a float for comparison
            plt.axvline(data, color=generate_random_color(), linestyle='--', label=f'{key} TT: {data:.3f}')
            
            # Calculate the p-value (sum of probabilities for data > data)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            probabilities = counts / sum(counts)
            p_value = sum(probabilities[bin_centers > data])
            
            # Add p-value to the plot
            random_height = random.uniform(0.4, 0.95) * max(counts)
            plt.text(data + 0.01, random_height, f'p-value = {p_value:.3f}', color='blue')
            
            # Determine the position (left tail, center, right tail)
            if data < bins[len(bins)//3]:
                position = "left tail"
            elif data > bins[2*len(bins)//3]:
                position = "right tail"
            else:
                position = "center"
            print(f"{key} profit {data:.3f} lies in the {position} of the distribution.")
            print(f"p-value: {p_value:.3f}")
            if p_value < 0.1:
                print("Since p-value < 0.1, we reject H0. The strategy likely outperformed that monkey.")
            else:
                print("Since p-value >= 0.1, we fail to reject H0. We can't beat that monkey.")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()