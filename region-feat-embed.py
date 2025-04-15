import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd

# Step 1: Data Loading
class BrainNetworkDataset(Dataset):
    def __init__(self, roi_dir, labels_file=None, transform=None):
        self.roi_dir = roi_dir
        self.transform = transform
        
        # Get all .1d files
        self.roi_files = sorted(glob.glob(os.path.join(roi_dir, "*.1d")))

        if labels_file is not None:
            try:
                self.region_networks = pd.read_csv(labels_file)
                print(f"Loaded region networks with columns: {self.region_networks.columns.tolist()}")
            except Exception as e:
                print(f"Error loading labels file: {e}")
                self.region_networks = None
    
    def __len__(self):
        return len(self.roi_files)
    
    def __getitem__(self, idx):
        # Load ROI time series data
        roi_file = self.roi_files[idx]
        subject_id = os.path.basename(roi_file).split('.')[0]  # ID from filename
        
        # time series data for each ROI
        roi_data = np.loadtxt(roi_file)  # Shape: [time_points, M] M is number of ROIs
        
        # Calculate Pearson correlation coefficient between ROIs to get adjacency matrix that's 200x200
        correlation_matrix = np.corrcoef(roi_data.T)  # Shape: [M, M]
        
        # Convert to tensor
        correlation_matrix = torch.FloatTensor(correlation_matrix)
        
        # Can make index the label
        label = idx
        
        sample = {'brain_network': correlation_matrix, 'label': label, 'subject_id': subject_id}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# Step 2: Region Feature Embedding Module Components
class RegionConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegionConvolution, self).__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, M, M]
        # Add channel dimension
        x = x.unsqueeze(1)  # [batch_size, 1, M, M]
        x = self.conv(x)  # [batch_size, out_channels, M, M]
        x = x.mean(dim=1)  # [batch_size, M, M]
        return x

class RegionEmbeddingMLP(nn.Module):
    def __init__(self, num_regions, hidden_dim):
        super(RegionEmbeddingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_regions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, M, M]
        batch_size, M, _ = x.size()
        
        # MLP to embed each row (connectivity profile of each ROI)
        x_reshaped = x.view(batch_size * M, M)
        embedded = self.mlp(x_reshaped)
        
        # Reshape back to somebody that i used to knowwww
        embedded = embedded.view(batch_size, M, -1)  # [batch_size, M, D]
        
        return embedded

# Step 3: Region Feature Embedding Module (WITHOUT MHSA)
class RegionFeatureEmbedding(nn.Module):
    def __init__(self, num_regions, hidden_dim, use_conv=True):
        super(RegionFeatureEmbedding, self).__init__()
        self.use_conv = use_conv
        
        if use_conv:
            self.region_conv = RegionConvolution(1, hidden_dim)
        
        self.region_embedding = RegionEmbeddingMLP(num_regions, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # when we do MHSA, we add here
        
    def forward(self, x):
        # x shape: [batch_size, M, M] - Correlation matrices
        
        if self.use_conv:
            x_conv = self.region_conv(x)
            x = x + x_conv  # Residual connection with convolution output
        
        # Region embedding using MLP
        x_reg = self.region_embedding(x)  # [batch_size, M, D]
        
        # Layer normalization
        x_norm = self.layer_norm(x_reg)
        
        return x_norm  # Pass this to MHSA

# Function to preprocess ROI data
def preprocess_roi_data(roi_dir, output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    roi_files = sorted(glob.glob(os.path.join(roi_dir, "*.1D")))
    processed_data = {}
    
    for roi_file in roi_files:
        subject_id = os.path.basename(roi_file).split('.')[0]
        try:
            roi_data = np.loadtxt(roi_file)
            correlation_matrix = np.corrcoef(roi_data.T)
            
            # Store in dictionary
            processed_data[subject_id] = correlation_matrix
            
            # if output_dir:
            #     output_file = os.path.join(output_dir, f"{subject_id}_corr.npy")
            #     np.save(output_file, correlation_matrix)
                
            # print(f"Processed {subject_id}: Matrix shape {correlation_matrix.shape}")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    return processed_data

def main():
    roi_dir = "/kaggle/input/autistic-brains/Outputs/cpac/nofilt_noglobal/rois_cc200"
    labels_file = "/kaggle/input/yeoyay/cc200_to_yeo7_mapping.csv"  # Yeo7 network assignments
    num_regions = 200  # for CC200 atlas
    hidden_dim = 64
    batch_size = 16
    
    try:
        if os.path.exists(roi_dir):
            roi_files = glob.glob(os.path.join(roi_dir, "*.1D"))
            print(f"Found {len(roi_files)} ROI files in {roi_dir}")
            
            # Print info about the first file
            if roi_files:
                first_file = roi_files[0]
                try:
                    data = np.loadtxt(first_file)
                    print(f"First file: {first_file}")
                    print(f"Data shape: {data.shape}")
                except Exception as e:
                    print(f"Error loading first file: {e}")
        else:
            print(f"ROI directory {roi_dir} not found")
            
        # Check the labels file
        if os.path.exists(labels_file):
            try:
                labels_df = pd.read_csv(labels_file)
                print(f"Labels file columns: {labels_df.columns.tolist()}")
                print(f"First few rows of labels file:\n{labels_df.head()}")
            except Exception as e:
                print(f"Error reading labels file: {e}")
        else:
            print(f"Labels file {labels_file} not found")
    except Exception as e:
        print(f"Error in data inspection: {e}")
    
    try:
        dataset = BrainNetworkDataset(roi_dir, labels_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = RegionFeatureEmbedding(num_regions, hidden_dim)
        
        for batch in dataloader:
            brain_networks = batch['brain_network']
            region_embeddings = model(brain_networks)
            print(f"Input shape: {brain_networks.shape}")
            print(f"Region embeddings shape: {region_embeddings.shape}")
            # break
    except Exception as e:
        print(f"Error in model execution: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        print("Starting data preprocessing...")
        processed_data = preprocess_roi_data(roi_dir, output_dir="processed_data")
        print(f"Preprocessed {len(processed_data)} subjects")
    except Exception as e:
        print(f"Error in preprocessing: {e}")

if __name__ == "__main__":
    main()
