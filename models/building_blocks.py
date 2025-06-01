import torch.nn as nn
import torch


def get_the_network_linear_list(output_sizes):
    '''
    Make FC linear layers with shapes defined by output_sizes list.
    '''
    linear_layers_list = nn.ModuleList()
    for i in range(len(output_sizes) - 1):
        linear_layers_list.append(nn.Linear(output_sizes[i], output_sizes[i + 1]))
        if i <len(output_sizes) - 2: linear_layers_list.append(nn.ReLU())
    return linear_layers_list

def forward_pass_linear_layer_relu(x, linear_layers_list):
    # Save original shape and flatten if needed
    original_shape = x.shape
    is_4d = len(original_shape) == 4
    
    if is_4d:
        # If 4D tensor, flatten to 2D
        batch_size, num_points, seq_len, feature_dim = original_shape
        x = x.reshape(-1, feature_dim)
    
    # Process through layers
    for layer in linear_layers_list:
        if isinstance(layer, nn.Linear):
            x = layer(x)
        elif isinstance(layer, nn.ReLU):
            x = layer(x)
        else:
            raise ValueError(f"Unexpected layer type: {type(layer)}")
    
    # Return in original shape if it was 4D
    if is_4d:
        x = x.reshape(batch_size, num_points, seq_len, -1)
    
    return x
