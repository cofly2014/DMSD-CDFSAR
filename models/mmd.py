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

def compute_rbf_kernel_1(x, y, sigma=1.0):
    """计算高斯核矩阵"""
    beta = 1.0 / (2.0 * sigma ** 2)


    dist = torch.cdist(x, y, p=2)
    return torch.exp(-beta * dist ** 2)

def compute_linear_kernel(x, y):

    num_samples_x = x.size(0)
    num_samples_y = y.size(0)

    # Reshape tensors to 2D for pairwise distance calculation
    x = x.view(num_samples_x, -1)  # Shape: (num_samples_x, feature_dim1 * feature_dim2)
    y = y.view(num_samples_y, -1)  # Shape: (num_samples_y, feature_dim1 * feature_dim2)

    """计算线性核矩阵"""
    return torch.mm(x, y.T)


def test(x, y):
    """计算高斯核矩阵"""

   # x= torch.mean(x, dim=0) #沿着size维度加
    #y = torch.mean(y, dim=0)  # 沿着size维度加
    dist = compute_l2_distance(x,y)
    return dist


def compute_l2_distance(x, y):
    """
    计算两个张量之间的L2距离
    Args:
        x: 张量，形状为 (num_samples_x, feature_dim)
        y: 张量，形状为 (num_samples_y, feature_dim)
    Returns:
        张量，表示每对向量的L2距离，形状为 (num_samples_x, num_samples_y)
    """
    dist = torch.cdist(x, y, p=2)
    return dist



def compute_rbf_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma ** 2)
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-beta * dist ** 2)

def compute_mmd(x, y, sigma=1.0):
    xx = compute_rbf_kernel(x, x, sigma)
    yy = compute_rbf_kernel(y, y, sigma)
    xy = compute_rbf_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
