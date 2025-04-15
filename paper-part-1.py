### OMFG THIS ONE WORKS
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd

# Data loading
class BrainNetworkDataset(Dataset):
    def __init__(self, roi_dir, labels_file=None, transform=None):
        self.roi_dir = roi_dir
        self.transform = transform
        
        # Get all .1D files
        self.roi_files = sorted(glob.glob(os.path.join(roi_dir, "*.1D")))
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

# Region Feature Embedding Module Components
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
            nn.ReLU(), # can change this later. My dimensions match, I am happy
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, M, M]
        batch_size, M, _ = x.size()
        
        # MLP to embed each row (connectivity profile of each ROI)
        x_reshaped = x.view(batch_size * M, M)
        embedded = self.mlp(x_reshaped)
        
        # Reshape back
        embedded = embedded.view(batch_size, M, -1)  # [batch_size, M, D]
        
        return embedded

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Query, Key, Value for each head
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, M, D] where M is number of regions, D is hidden dimension
        batch_size, M, D = x.size()
        
        # Linear projections and reshape for multi-head attention
        # Used formula (3), (4), (5) from paper
        q = self.q_linear(x).view(batch_size, M, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, M, head_dim]
        k = self.k_linear(x).view(batch_size, M, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, M, head_dim]
        v = self.v_linear(x).view(batch_size, M, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, M, head_dim]
        
        # Scaled dot-product attention
        # Formula (6) from paper
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, M, M]
        
        # Diagonal masking to avoid self-ROI relations
        # Formula (7) from the paper
        mask = torch.ones_like(scores)
        mask = mask.triu(diagonal=1) + mask.tril(diagonal=-1)  # mask 1s everywhere except diagonal
        scores = scores * mask  # apply the mask and set diagonal to 0
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, M, M]
        attn_weights = self.dropout(attn_weights)
        
        # apply attention weights to values (formula (8) & (9) from paper)
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, M, head_dim]
        
        # Reshape and concatenate heads THIS IS WHERE I WAS GOING WRONG
        context = context.transpose(1, 2).contiguous().view(batch_size, M, D)  # [batch_size, M, D]
        
        # Final linear projection
        output = self.out_proj(context)  # [batch_size, M, D]
        
        return output

class RegionFeatureEmbedding(nn.Module):
    def __init__(self, num_regions, hidden_dim, num_heads=8, use_conv=True):
        super(RegionFeatureEmbedding, self).__init__()
        self.use_conv = use_conv
        
        # Region convolution component
        if use_conv:
            self.region_conv = RegionConvolution(1, hidden_dim)
        
        # Region embedding MLP
        self.region_embedding = RegionEmbeddingMLP(num_regions, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-head self-attention
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads)
        
        # Output projection as mentioned in the paper (W_reg,O)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch_size, M, M] - Correlation matrices
        
        # Apply region convolution if enabled
        if self.use_conv:
            x_conv = self.region_conv(x)
            x = x + x_conv  # Residual connection with convolution output
        
        # Region embedding using MLP to get X_reg 
        x_reg = self.region_embedding(x)  # [batch_size, M, D]
        
        # Layer normalization before MHSA
        x_norm = self.layer_norm(x_reg)
        
        # Apply multi-head self-attention
        # Formula (1) from the paper (directly use the output f layernorm)
        attn_output = self.mhsa(x_norm)  # [batch_size, M, D]
        
        # Apply output projection
        attn_output = self.output_proj(attn_output)
        
        # Add residual connection (formula (10) from paper)
        output = attn_output + x_reg  # [batch_size, M, D] # does this need to be fixed, though????
        
        return output

class SubnetworkFeatureEmbedding(nn.Module):
    def __init__(self, num_subnets, region_per_subnet, hidden_dim, num_heads=8):
        super(SubnetworkFeatureEmbedding, self).__init__()
        self.num_subnets = num_subnets  # R = 7 for Yeo atlas
        self.region_per_subnet = region_per_subnet  # number of regions per subnet
        
        # MLP for each subnetwork
        self.subnet_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(regions * regions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for regions in region_per_subnet
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-head self-attention for subnetworks
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, subnet_masks):
        # x shape: [batch_size, M, M]
        # subnet_masks: List of masks for each subnetwork
        batch_size = x.size(0)
        
        # Process each subnetwork
        subnet_features = []
        
        for i in range(self.num_subnets):
            # Extract subnetwork using mask
            subnet_mask = subnet_masks[i].to(x.device)
            
            # Get the indices where mask is 1
            indices = subnet_mask.nonzero()
            
            # Extract the ROIs for this subnetwork
            roi_indices = torch.unique(indices[:, 0])
            num_subnet_regions = len(roi_indices)
            
            # Extract the subnetwork connectivity matrix
            subnet = x[:, roi_indices][:, :, roi_indices]  # [batch_size, subnet_size, subnet_size]
            
            # Flatten the subnetwork
            subnet_flat = subnet.reshape(batch_size, -1)  # [batch_size, subnet_size^2]
            
            # Apply MLP
            subnet_embedding = self.subnet_mlps[i](subnet_flat)  # [batch_size, D]
            subnet_features.append(subnet_embedding)
        
        # Stack subnetwork features
        x_sub = torch.stack(subnet_features, dim=1)  # [batch_size, R, D]
        
        # Layer normalization
        x_norm = self.layer_norm(x_sub)
        
        # Apply multi-head self-attention
        attn_output = self.mhsa(x_norm)  # [batch_size, R, D]
        
        # Apply output projection
        attn_output = self.output_proj(attn_output)
        
        # Add residual connection (formula (11) from the paper)
        output = attn_output + x_sub  # [batch_size, R, D]
        
        return output

# create subnetwork masks based on Yeo atlas mapping - FIXED VERSION (i hate this part with all my life. thank god for gpt)
def create_yeo_subnet_masks(roi_to_yeo_mapping, num_regions=200, num_subnets=7):
    """
    Create masks for each Yeo subnetwork based on the ROI to Yeo mapping.
    
    Parameters:
    - roi_to_yeo_mapping: DataFrame with mapping between ROIs and Yeo networks
    - num_regions: Total number of ROIs
    - num_subnets: Number of Yeo subnetworks
    
    Returns:
    - subnet_masks: List of masks for each subnetwork
    - regions_per_subnet: List containing number of regions in each subnet
    """
    # Initialize masks
    subnet_masks = []
    regions_per_subnet = []
    
    # For each Yeo network (1-7)
    for subnet_id in range(1, num_subnets + 1):
        # Get regions belonging to this network
        subnet_regions = roi_to_yeo_mapping[roi_to_yeo_mapping['Yeo7_Network'] == subnet_id]['CC200_Region'].values
        
        # Count regions in this subnet
        num_subnet_regions = len(subnet_regions)
        regions_per_subnet.append(num_subnet_regions)
        
        # Create mask (indices start from 1 in the CSV but 0 in Python)
        mask = torch.zeros((num_regions, num_regions))
        
        # Convert to 0-indexed
        subnet_regions_0indexed = [r-1 for r in subnet_regions]
        
        # Set mask values
        for i in subnet_regions_0indexed:
            for j in subnet_regions_0indexed:
                mask[i, j] = 1.0
                
        subnet_masks.append(mask)
    
    return subnet_masks, regions_per_subnet

# same as before
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
                
            # print(f"Processed {subject_id}: Matrix shape {correlation_matrix.shape}")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    return processed_data

# the first part of the hhgraphfromer
class HHGraphFormer(nn.Module):
    def __init__(self, num_regions, num_subnets, region_per_subnet, hidden_dim, num_heads=8, num_classes=2):
        super(HHGraphFormer, self).__init__()
        
        # Region feature embedding module
        self.region_feature_embedding = RegionFeatureEmbedding(num_regions, hidden_dim, num_heads)
        
        # Subnetwork feature embedding module
        self.subnetwork_feature_embedding = SubnetworkFeatureEmbedding(num_subnets, region_per_subnet, hidden_dim, num_heads)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_regions + num_subnets), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, subnet_masks):
        # x shape: [batch_size, M, M]
        batch_size = x.size(0)
        
        # Region feature embedding
        region_embeddings = self.region_feature_embedding(x)  # [batch_size, M, D]
        
        # Subnetwork feature embedding
        subnet_embeddings = self.subnetwork_feature_embedding(x, subnet_masks)  # [batch_size, R, D]
        
        # Flatten embeddings
        region_flat = region_embeddings.view(batch_size, -1)  # [batch_size, M*D]
        subnet_flat = subnet_embeddings.view(batch_size, -1)  # [batch_size, R*D]
        
        # Concatenate for classification
        combined = torch.cat([region_flat, subnet_flat], dim=1)  # [batch_size, (M+R)*D]
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

def main():
    roi_dir = "/kaggle/input/autistic-brains/Outputs/cpac/nofilt_noglobal/rois_cc200"
    labels_file = "/kaggle/input/yeoyay/cc200_to_yeo7_mapping.csv"  # Yeo7 network assignments
    num_regions = 200  # for CC200 atlas
    hidden_dim = 64
    num_heads = 8
    num_subnets = 7  # 7 networks in Yeo atlas
    batch_size = 16
    
    # Default values for subnet masks in case of errors
    regions_per_subnet = [int(num_regions/num_subnets)] * num_subnets
    subnet_masks = [torch.ones((num_regions, num_regions))/num_subnets for _ in range(num_subnets)]
    
    try:
        if os.path.exists(roi_dir):
            roi_files = glob.glob(os.path.join(roi_dir, "*.1D"))
            print(f"Found {len(roi_files)} ROI files in {roi_dir}")
            
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
            
        if os.path.exists(labels_file):
            try:
                labels_df = pd.read_csv(labels_file)
                print(f"Labels file columns: {labels_df.columns.tolist()}")
                print(f"First few rows of labels file:\n{labels_df.head()}")
                
                subnet_masks, regions_per_subnet = create_yeo_subnet_masks(labels_df, num_regions, num_subnets)
                print(f"Created {len(subnet_masks)} subnet masks")
                print(f"Regions per subnet: {regions_per_subnet}")
            except Exception as e:
                print(f"Error reading labels file: {e}")
        else:
            print(f"Labels file {labels_file} not found")
    except Exception as e:
        print(f"Error in data inspection: {e}")
    
    try:
        dataset = BrainNetworkDataset(roi_dir, labels_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        region_model = RegionFeatureEmbedding(num_regions, hidden_dim, num_heads)
        
        subnet_model = SubnetworkFeatureEmbedding(num_subnets, regions_per_subnet, hidden_dim, num_heads)
        
        for batch in dataloader:
            brain_networks = batch['brain_network']
            
            # Process with region feature embedding
            region_embeddings = region_model(brain_networks)
            print(f"Input shape: {brain_networks.shape}")
            print(f"Region embeddings shape: {region_embeddings.shape}")
            
            # Process with subnetwork feature embedding
            subnet_embeddings = subnet_model(brain_networks, subnet_masks)
            print(f"Subnetwork embeddings shape: {subnet_embeddings.shape}")
            break
            
        # Initialize and test the full model
        full_model = HHGraphFormer(num_regions, num_subnets, regions_per_subnet, hidden_dim, num_heads, num_classes=2)
        
        for batch in dataloader:
            brain_networks = batch['brain_network']
            outputs = full_model(brain_networks, subnet_masks)
            print(f"Full model output shape: {outputs.shape}")
            break
            
    except Exception as e:
        print(f"Error in model execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


'''
- Found 408 ROI files from the dataset
- Confirmed the first file's shape is (296, 200), which means 296 time points for 200 ROIs
- Successfully loaded the Yeo7 network mapping file
- Created 7 subnet masks (one for each Yeo network)
- Correctly identified the number of regions per subnet: [43, 22, 21, 22, 15, 21, 41]
- some ROIs have zero variance (no signal variation) that's why the weird zero error thing. Idt it breaks the pipeline- check out once
- Input shape: [16, 200, 200] - 16 subjects, each with a 200×200 correlation matrix
- Region embeddings shape: [16, 200, 64] - each of the 200 regions now has a 64-dimensional embedding
- Subnetwork embeddings shape: [16, 7, 64] - each of the 7 subnetworks now has a 64-dimensional embedding
- Full model output shape: [16, 2] - final classification output with 2 classes (control vs. ASD)
'''

'''
Output looks like this:

Input shape: torch.Size([16, 200, 200])
Region embeddings shape: torch.Size([16, 200, 64])
Subnetwork embeddings shape: torch.Size([16, 7, 64])
Full model output shape: torch.Size([16, 2])
'''
