import random
import numpy as np
import torch
from skimage import morphology

from utils.losses import soft_skel

def calculate_angle(branch_a, branch_b):
    # Calculate differences in x and y
    diff_x = branch_b[1] - branch_a[1]  # Difference in x-coordinates
    diff_y = branch_b[0] - branch_a[0]  # Difference in y-coordinates
    
    # Calculate angle in degrees using arctan2, even with scalar inputs
    angle = np.arctan2(diff_y, diff_x) * (180.0 / np.pi)
    
    return np.abs(angle)

def select_thin_branch(branches):
    min_angle_sum = float('inf')
    thin_branch = None
    
    for i in range(len(branches)):
        # Take the two neighboring branches
        prev_branch = branches[i - 1]
        next_branch = branches[(i + 1) % len(branches)]
        
        angle_sum = calculate_angle(branches[i], prev_branch) + calculate_angle(branches[i], next_branch)
        
        if angle_sum < min_angle_sum:
            min_angle_sum = angle_sum
            thin_branch = branches[i]
    
    return thin_branch

def get_branch_points(skeleton):
    # Calculate branch points using a convolution kernel
    kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=skeleton.device).view(1, 1, 3, 3)
    branch_points = torch.nn.functional.conv2d(skeleton, kernel, padding=1) * skeleton
    return (branch_points > 2).float()  # Threshold to get branch points

def get_neighboring_branches(skeleton, point):
    # Define the order of neighbors in a clockwise direction
    neighbor_offsets = [
        (-1, 0),  # Up
        (-1, 1),  # Up-Right
        (0, 1),   # Right
        (1, 1),   # Down-Right
        (1, 0),   # Down
        (1, -1),  # Down-Left
        (0, -1),  # Left
        (-1, -1)  # Up-Left
    ]
    
    neighbors = []
    
    for dy, dx in neighbor_offsets:
        neighbor = (point[0] + dy, point[1] + dx)
        if is_valid_coordinate(neighbor, skeleton) and skeleton[..., neighbor[0], neighbor[1]] > 0:
            neighbors.append(neighbor)
    
    return neighbors

def is_valid_coordinate(coord, tensor):
    return 0 <= coord[0] < tensor.size(-2) and 0 <= coord[1] < tensor.size(-1)

def get_direction(branch, branch_point):
    # Calculate the direction from the branch to the branch point
    angle = calculate_angle(branch, branch_point)
    angle_coor = np.array([np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)])  # Convert to radians
    return np.around(angle_coor).astype(int)

def skeletonize_tensor(tensor):
    # Ensure the tensor is on the CPU and convert to NumPy
    tar = tensor.squeeze(1).cpu().numpy()  # shape n*h*w
    skeletons = []

    for i in range(tar.shape[0]):
        # Convert to binary image
        binary_image = (tar[i] > 0)  # Convert to boolean (0 and 1)
        
        # Skeletonization
        skeleton = morphology.skeletonize(binary_image)
        skeletons.append(skeleton)

    # Stack back into a tensor
    skeleton_tensor = torch.tensor(np.array(skeletons)).float().unsqueeze(1).to(tensor.device)  # shape n*1*h*w
    return skeleton_tensor

def branch_based_invasion(tar_tensor: torch.Tensor, erosion_radius: int=50, offset: int=0, circle_mode=True):
    skeleton = skeletonize_tensor(tar_tensor)
    branch_points = get_branch_points(skeleton)

    # Create a tensor to indicate whether invasion occurred
    invasion_occurred = torch.ones(tar_tensor.size(0), dtype=torch.float32, device=tar_tensor.device)

    for idx in range(tar_tensor.size(0)):
        if branch_points[idx].sum() == 0:
            invasion_occurred[idx] = 0
            continue

        # Randomly select a branch point
        branch_point = torch.nonzero(branch_points[idx])[random.randint(0, int(branch_points[idx].sum().item() - 1))].squeeze().tolist()
        
        # Get neighboring branches
        branches = get_neighboring_branches(skeleton[idx], branch_point[-2:])

        # Select the thinnest branch
        thin_branch = select_thin_branch(branches)

        # Get the invasion direction
        direction = get_direction(thin_branch, branch_point)

        # Calculate absolute coordinates based on the direction
        y, x = torch.meshgrid(torch.arange(-erosion_radius, erosion_radius+1), torch.arange(-erosion_radius, erosion_radius+1), indexing='ij')
        invasion_y = branch_point[-2] + y + direction[0] * (erosion_radius + offset)
        invasion_x = branch_point[-1] + x + direction[1] * (erosion_radius + offset)

        if circle_mode:
            # Create a mask for points within the circular area
            distances = (y ** 2 + x ** 2) <= erosion_radius ** 2  # Points within the radius
        else:
            # If not circle mode, simply consider the rectangle
            distances = torch.ones_like(y, dtype=torch.bool)  # All points valid in rectangular mode

        # Check if invasion coordinates are within the image boundaries
        invasion_y = torch.clamp(invasion_y, 0, tar_tensor.size(-2) - 1)
        invasion_x = torch.clamp(invasion_x, 0, tar_tensor.size(-1) - 1)

        # Apply the erosion only to points within the valid mask
        valid_y = invasion_y[distances]
        valid_x = invasion_x[distances]

        tar_tensor[idx, 0, valid_y, valid_x] = 0  # Set invaded region to zero

    return tar_tensor, invasion_occurred  # Return the modified tensor and invasion indicator

def rotate_patch(patch, angle):
    """Rotate the patch by a given angle."""
    # Using pytorch's grid_sample to rotate the patch
    theta = torch.tensor([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]
    ], dtype=torch.float).unsqueeze(0).to(patch.device)
    
    grid = torch.nn.functional.affine_grid(theta, patch.size(), align_corners=False)
    rotated_patch = torch.nn.functional.grid_sample(patch, grid, mode='bilinear', align_corners=False)
    
    return rotated_patch

def random_over_invasion(tar, mode='center', radius=10):
    state = torch.ones(tar.size(0), dtype=torch.float32, device=tar.device)  # For returning whether invasion occurred per sample
    
    for idx in range(tar.size(0)):  # Loop over batch
        # Find all positions in tar where value is > 0
        valid_positions = torch.nonzero(tar[idx, 0] > 0, as_tuple=False)
        
        if valid_positions.size(0) == 0:
            state[idx] = 0  # No valid positions to process, set state to 0
            continue
        
        # Randomly select a point with value > 0
        selected_point = valid_positions[random.randint(0, valid_positions.size(0) - 1)]
        y, x = selected_point.tolist()

        # Determine the region to copy, considering the mode (center or corner)
        if mode == 'center':
            y_start = max(0, y - radius)
            x_start = max(0, x - radius)
            y_end = min(tar.size(-2), y + radius + 1)
            x_end = min(tar.size(-1), x + radius + 1)
        elif mode == 'corner':
            y_start = y
            x_start = x
            y_end = min(tar.size(-2), y + 2 * radius + 1)
            x_end = min(tar.size(-1), x + 2 * radius + 1)

        # Extract the patch
        patch = tar[idx:idx+1, :, y_start:y_end, x_start:x_end].clone()

        # Rotate the patch 90 degrees
        rotated_patch = rotate_patch(patch, angle=90)

        # Overlay the rotated patch back to the original tensor at the same location
        tar[idx, :, y_start:y_end, x_start:x_end] += rotated_patch.squeeze(0)
    
    tar = torch.clamp(tar, 0, 1)  # Clamp the tensor values to be within [0, 1]

    if mode == 'center':
        state = 1 - state
    elif mode == 'corner':
        state = state * 0
    return tar, state  

def random_dilate(tensor: torch.Tensor, iter=1):
    for _ in range(iter):
        tensor = torch.nn.functional.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    return tensor
