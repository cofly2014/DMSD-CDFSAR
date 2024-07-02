import torch


def compute_rbf_kernel(x, y, sigma):
    """
    Computes the RBF (Gaussian) kernel between two sets of samples with a given sigma.
    Args:
        x: Tensor of shape (num_samples_x, feature_dim1, feature_dim2).
        y: Tensor of shape (num_samples_y, feature_dim1, feature_dim2).
        sigma: Bandwidth parameter for the RBF kernel.
    Returns:
        Tensor of shape (num_samples_x, num_samples_y) representing the RBF kernel matrix.
    """
    # Reshape input tensors for pairwise distance calculation
    num_samples_x = x.size(0)
    num_samples_y = y.size(0)

    # Reshape tensors to 2D for pairwise distance calculation
    x = x.view(num_samples_x, -1)  # Shape: (num_samples_x, feature_dim1 * feature_dim2)
    y = y.view(num_samples_y, -1)  # Shape: (num_samples_y, feature_dim1 * feature_dim2)

    # Compute pairwise squared Euclidean distances
    dist = torch.cdist(x, y, p=2).pow(2)  # Shape: (num_samples_x, num_samples_y)

    # Compute RBF kernel
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel


def compute_mmd(x, y, sigma):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples using the RBF kernel.
    Args:
        x: Tensor of shape (num_samples_x, feature_dim1, feature_dim2).
        y: Tensor of shape (num_samples_y, feature_dim1, feature_dim2).
        sigma: Bandwidth parameter for the RBF kernel.
    Returns:
        Scalar representing the MMD value.
    """
    xx = compute_rbf_kernel(x, x, sigma)
    yy = compute_rbf_kernel(y, y, sigma)
    xy = compute_rbf_kernel(x, y, sigma)

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


# Example usage
num_samples = 256
feature_dim1 = 32
feature_dim2 = 32

# Use a small sigma value to increase sensitivity to differences
sigma = 0.1

# Generate random input data multiple times to check MMD results
mmd_results = []
for _ in range(5):
    x = torch.randn(num_samples, feature_dim1, feature_dim2)  # Source domain samples
    y = torch.randn(num_samples, feature_dim1, feature_dim2)  # Target domain samples

    mmd_value = compute_mmd(x, y, sigma)
    mmd_results.append(mmd_value.item())
    print('MMD with random data:', mmd_value.item())

# Print all results to compare
print('All MMD results:', mmd_results)