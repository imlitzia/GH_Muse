import csv
import numpy as np
import matplotlib.pyplot as plt

# Load data from the CSV file
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get header
        
        # Find indices for the columns we need
        beta_af7_idx = header.index('Beta_AF7')
        beta_af8_idx = header.index('Beta_AF8')
        alpha_af7_idx = header.index('Alpha_AF7')
        alpha_af8_idx = header.index('Alpha_AF8')
        
        data = []
        ratios_af7 = []
        ratios_af8 = []
        
        for row in reader:
            # Skip event marker rows (they start with '/muse')
            if any(cell.startswith('/muse') for cell in row):
                continue
                
            try:
                # Check if row has enough columns and contains numeric data
                if len(row) >= max(beta_af7_idx, beta_af8_idx, alpha_af7_idx, alpha_af8_idx):
                    # Get the relevant values
                    alpha_af7 = float(row[alpha_af7_idx]) if row[alpha_af7_idx].strip() else 0.0
                    beta_af7 = float(row[beta_af7_idx]) if row[beta_af7_idx].strip() else 0.0
                    alpha_af8 = float(row[alpha_af8_idx]) if row[alpha_af8_idx].strip() else 0.0
                    beta_af8 = float(row[beta_af8_idx]) if row[beta_af8_idx].strip() else 0.0
                    
                    # Calculate ratios
                    ratio_af7 = beta_af7 / alpha_af7 if alpha_af7 != 0 else 0
                    ratio_af8 = beta_af8 / alpha_af8 if alpha_af8 != 0 else 0
                    
                    # Only append if we have valid data
                    if not (alpha_af7 == 0 and beta_af7 == 0 and alpha_af8 == 0 and beta_af8 == 0):
                        ratios_af7.append(ratio_af7)
                        ratios_af8.append(ratio_af8)
                        data.append([alpha_af7, beta_af7, alpha_af8, beta_af8])
                
            except ValueError:
                # Skip rows that can't be converted to float
                continue
    
    if not data:
        raise ValueError("No valid data found in the CSV file")
    
    # Calculate averages of the ratios
    avg_ratio_af7 = np.mean(ratios_af7)
    avg_ratio_af8 = np.mean(ratios_af8)
    
    print(f"Average Beta/Alpha Ratio for AF7: {avg_ratio_af7:.3f}")
    print(f"Average Beta/Alpha Ratio for AF8: {avg_ratio_af8:.3f}")
    
    return np.array(data), np.array(ratios_af7), np.array(ratios_af8)

# Visualize brainwave data
def visualize_data(data, ratios_af7, ratios_af8, frequency_labels):
    time = np.arange(len(ratios_af7))  # Time axis

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Brainwave Analysis')

    # Plot Alpha and Beta values
    ax1.plot(time, data[:, 0], label='Alpha AF7', alpha=0.7)
    ax1.plot(time, data[:, 1], label='Beta AF7', alpha=0.7)
    ax1.plot(time, data[:, 2], label='Alpha AF8', alpha=0.7)
    ax1.plot(time, data[:, 3], label='Beta AF8', alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.set_title('Alpha and Beta Values')

    # Plot Beta/Alpha ratios
    ax2.plot(time, ratios_af7, label='AF7 Beta/Alpha', alpha=0.7)
    ax2.plot(time, ratios_af8, label='AF8 Beta/Alpha', alpha=0.7)
    ax2.set_ylabel('Beta/Alpha Ratio')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.set_title('Beta/Alpha Ratios')
    
    plt.tight_layout()
    plt.show()

# Example usage
filename = 'brainwave_data.csv'  # Replace with your CSV filename
brainwave_data, ratios_af7, ratios_af8 = load_data(filename)
frequency_labels = [0, 4, 8, 12, 16, 20, 24]  # Replace with your frequency labels if available
visualize_data(brainwave_data, ratios_af7, ratios_af8, frequency_labels)
