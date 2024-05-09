import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum
from scipy.ndimage import zoom


class Color(Enum):
    NO_COLOR = 0
    GREEN = 1
    WHITE = 2
    ORANGE = 3
    YELLOW = 4

def parse_pcd_data(data):
    # Split the data into lines (PCD header and point data)
    lines = data.split(b'\n')

    # Extract metadata from the PCD header (e.g., number of points)
    num_points = int(lines[9].split()[-1])  # Assuming line 9 contains this information

    # Initialize an empty NumPy array to store the points
    points = np.zeros((num_points, 3), dtype=np.float32)

    # Parse the point data (assuming XYZ format, with one point per line)
    for i in range(num_points):
        x, y, z = map(float, lines[i + 11].split())  # Assuming point data starts from line 11
        points[i] = [x, y, z]

    return points

def get_obstacle_map_from_pcd(pcd_filename, grid_size):
    """
    :param pcd_filename: relative filename of the pointcloud
    :param grid_size: size of grid in meters
    :return: obstacle map with given grid size
    """
    """
    # Open the point cloud file in binary mode
    with open(pcd_filename, 'rb') as file:
        # Read the data from the file
        data = file.read()

    points = parse_pcd_data(data)
    """


    # Load the point cloud from file
    points = np.loadtxt(pcd_filename)


    # Calculate the maximum and minimum x and y values
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])

    # Calculate the number of rows and columns in the grid
    num_rows = int(np.ceil((y_max - y_min) / grid_size))
    num_cols = int(np.ceil((x_max - x_min) / grid_size))

    # Create the grid
    obstacle_map = np.zeros((num_rows, num_cols))

    # Iterate over the points and fill in the grid
    for point in points:
        x_index = int(np.floor((point[0] - x_min) / grid_size))
        y_index = int(np.floor((point[1] - y_min) / grid_size))
        if (point[2] > 5):
            obstacle_map[y_index, x_index] = max(obstacle_map[y_index, x_index], round(point[2]))

    # TODO: Create coordinates map that stores the relative coordinates of points
    return obstacle_map

def center_nonzero_data(obstacle_map):
    # Find coordinates of non-zero elements
    y, x = np.where(obstacle_map != 0)

    # Find min and max coordinates for non-zero elements
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)

    # Calculate the center of mass of the bounding box
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2

    # Calculate the center of the entire map
    y_map_center = obstacle_map.shape[0] // 2
    x_map_center = obstacle_map.shape[1] // 2

    # Calculate the offset required to center the non-zero data
    y_offset = y_map_center - y_center
    x_offset = x_map_center - x_center

    # Create a new obstacle map of the same shape as the original
    centered_obstacle_map = np.zeros_like(obstacle_map)

    # Translate the non-zero data to the new center
    for i, j in zip(y, x):
        new_i, new_j = i + y_offset, j + x_offset

        # Check if the new coordinates are within the bounds of the array
        if (0 <= new_i < centered_obstacle_map.shape[0]) and (0 <= new_j < centered_obstacle_map.shape[1]):
            centered_obstacle_map[new_i, new_j] = obstacle_map[i, j]

    return centered_obstacle_map

def display_map(grid_map, figsize=(8, 8)):
    """
    :param grid_map: The grid map to display.
    :param figsize: The size of the figure in inches (width, height).
    """
    # Create a larger figure with the specified size
    plt.figure(figsize=figsize)

    # Plot the grid as an image
    plt.imshow(grid_map, cmap='gray')

    # Show the plot
    plt.show()

def resize_map(obstacle_map, new_shape):
    y_scale = new_shape[0] / obstacle_map.shape[0]
    x_scale = new_shape[1] / obstacle_map.shape[1]
    
    # Use scipy's zoom to resize the array
    resized_map = zoom(obstacle_map, (y_scale, x_scale))
    
    return resized_map

def manual_resize_map(obstacle_map, new_shape):
    old_shape = obstacle_map.shape
    y_ratio, x_ratio = old_shape[0] // new_shape[0], old_shape[1] // new_shape[1]
    
    resized_map = np.zeros(new_shape)
    
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # Compute the average for this cell
            subarray = obstacle_map[i*y_ratio:(i+1)*y_ratio, j*x_ratio:(j+1)*x_ratio]
            resized_map[i, j] = np.mean(subarray)
            
    return resized_map

def save_png_from_numpy(numpy_array, output_img_filename):
    # convert the numpy array to PIL Image
    img = Image.fromarray((numpy_array * 255).astype(np.uint8), mode='L')

    # save the image as PNG file
    img.save(output_img_filename)


def save_numpy_as_npy(numpy_array, output_filename):
    # save the numpy array in the output_filename
    np.save(output_filename, numpy_array)
