import pandas as pd
import numpy as np

def convert_csv_to_npy(csv_path, npy_path):
    print(f"Loading {csv_path}...")
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    data_list = []
    
    # Change 'pid' if your participant ID column has a different name
    # (e.g., 'subject_id', 'ID', etc.)
    pid_column = 'pid'
    
    print("Processing rows...")
    for idx, row in df.iterrows():
        # 1. Extract the participant ID
        pid = int(row[pid_column])
        
        # 2. Extract the PPG sequence
        # We drop the PID column so only the 1200 signal values remain
        ppg_signal = row.drop(labels=[pid_column]).values.astype(np.float32)
        
        # Check to ensure the sequence length matches our expectations
        if len(ppg_signal) != 1200:
            print(f"Warning: Row {idx} has length {len(ppg_signal)} instead of 1200.")
            
        # 3. Append to our list as a dictionary
        data_list.append({
            "ppg": ppg_signal,
            "pid": pid
        })

    # 4. Save the list of dictionaries as an .npy file
    # allow_pickle=True is required when saving lists of dictionaries in numpy
    np.save(npy_path, data_list, allow_pickle=True)
    print(f"Successfully saved {len(data_list)} records to {npy_path}")

if __name__ == "__main__":
    # Update these paths to point to your actual files
    INPUT_CSV = "path/to/your/rasouli_data.csv"
    OUTPUT_NPY = "rasouli_ppg.npy"
    
    convert_csv_to_npy(INPUT_CSV, OUTPUT_NPY)