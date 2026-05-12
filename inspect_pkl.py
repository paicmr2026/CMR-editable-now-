import torch
import os

# Set the path to your file
file_path = './experiments/cub/embeddings/cub/train_x.pt'

if os.path.exists(file_path):
    # Load the tensor
    y_train = torch.load(file_path)
    
    print(f"--- Inspection of {os.path.basename(file_path)} ---")
    print(f"Type: {type(y_train)}")
    
    # Check if it's a standard tensor
    if torch.is_tensor(y_train):
        print(f"Shape: {y_train.shape}")
        print(f"Dtype: {y_train.dtype}")
        print(f"First 5 elements:\n{y_train[:5]}")
        
        # Check if it contains class indices or one-hot vectors
        if y_train.ndimension() == 1:
            print("Status: 1D tensor detected (likely class indices).")
        else:
            print(f"Status: {y_train.ndimension()}D tensor detected (likely one-hot or multi-label).")
    else:
        print("The file contains a non-tensor object (e.g., a list or dictionary).")
else:
    print(f"File not found at {file_path}")