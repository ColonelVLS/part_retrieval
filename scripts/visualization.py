import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

#20 different colors
palette = np.array([
    [0.65098039, 0.80784314, 0.89019608],
    [0.89019608, 0.10196078, 0.10980392],
    [0.69803922, 0.8745098 , 0.54117647],
    [0.2       , 0.62745098, 0.17254902],
    [0.98431373, 0.60392157, 0.6       ],
    [1.        , 1.        , 0.6       ],
    [0.99215686, 0.74901961, 0.43529412],
    [1.        , 0.49803922, 0.        ],
    [0.79215686, 0.69803922, 0.83921569],
    [0.41568627, 0.23921569, 0.60392157],
    [0.12156863, 0.47058824, 0.70588235],
    [0.69411765, 0.34901961, 0.15686275],
    [0.97254902, 0.97254902, 0.52941176],
    [0.78039216, 0.78039216, 0.78039216],
    [0.7372549 , 0.74117647, 0.13333333],
    [0.85882353, 0.85882353, 0.55294118],
    [0.09019608, 0.74509804, 0.81176471],
    [0.61960784, 0.85490196, 0.89803922],
    [0.99215686, 0.85490196, 0.9254902 ],
    [1.        , 0.92941176, 0.43529412]
])

colormaps = {
    "viridis": cm.viridis,
    "plasma": cm.plasma,
    "inferno": cm.inferno,
    "magma": cm.magma,
    "cividis": cm.cividis
}

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

def quick_vis(x, extra_geometries = [], title=None):

    assert isinstance(extra_geometries, list)
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    o3d.visualization.draw_geometries([
        o3d.geometry.PointCloud(
            points = o3d.utility.Vector3dVector(x)
        )
    ] + extra_geometries, window_name=title if title is not None else "Open3D")

def quick_vis_pretty(x, colormap = "viridis", extra_geometries = [], title = None):

    assert isinstance(extra_geometries, list)
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    #normalizing the scale to unit sphere
    x = x / np.linalg.norm(x, axis=1).max()

    #creating a depth based colormap
    normalized_zs = (x[:, 2] - np.min(x[:, 2])) / (np.max(x[:, 2]) - np.min(x[:, 2]))
    cmap = colormaps[colormap]
    colors = cmap(normalized_zs)[:,:-1]

    #creating the spheres that will represent the points
    spheres = o3d.geometry.TriangleMesh()

    for i, p in enumerate(x):
        spheres += o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5).translate(p).paint_uniform_color(colors[i])
        
    #visualizing    
    o3d.visualization.draw_geometries([spheres] + extra_geometries, window_name=title if title is not None else "Open3D")

def quick_vis_with_parts(x, parts, extra_geometries = [], title=None):

    if isinstance(parts, torch.Tensor):
        parts = parts.cpu().numpy()
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    num_parts = np.max(parts)
    palette = torch.rand(np.max(parts), 3).numpy()
    colors = np.zeros_like(x)

    for i in range(num_parts):
        colors[parts == i] = palette[i,:]
    
    point_cloud = o3d.geometry.PointCloud(
            points = o3d.utility.Vector3dVector(x),
        )

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([
        point_cloud
    ] + extra_geometries, window_name=title if title is not None else "Open3D")

def quick_vis_with_parts_pretty(x, parts, extra_geometries = [], title=None):

    if isinstance(parts, torch.Tensor):
        parts = parts.cpu().detach().numpy()
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    #normalizing the scale to unit sphere
    x = x / np.linalg.norm(x, axis=1).max()

    num_parts = np.max(parts)
    assert num_parts <= len(palette), "Number of parts exceeds the number of colors in the palette"
    
    spheres = o3d.geometry.TriangleMesh()

    #creating the spheres that will represent the points
    for i, (p, pid) in enumerate(zip(x, parts)):

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5).translate(p)
        sphere.paint_uniform_color(palette[pid])
        spheres += sphere

    o3d.visualization.draw_geometries([
        spheres
    ] + extra_geometries, window_name=title if title is not None else "Open3D")

def quick_vis_with_parts_many(x, parts, xy, title=None):

    def generate_3d_grid(rows, columns):
        # Create arrays of x and y coordinates
        x_coords = np.arange(columns) * 2
        y_coords = np.arange(rows) * 2
        
        # Use meshgrid to create grid of x and y coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Stack the x, y, and z coordinates to form the 3D points
        points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()])        
        points = points - np.mean(points, axis=0)

        return points

    #Sanity checks
    if isinstance(parts, torch.Tensor):
        parts = parts.cpu().numpy()
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    assert len(x.shape) == 3 and len(parts.shape) == 2
    assert x.shape[0] == parts.shape[0] and x.shape[1] == parts.shape[1]

    #keeping a list of the geometries
    geometries = []

    #centers upon each point cloud will be placed
    centers = generate_3d_grid(xy[0], xy[1])

    #generating a color palette to use for the different parts
    palette = torch.rand(np.max(parts), 3).numpy()
    
    for i in range(x.shape[0]):    
        
        num_parts = np.max(parts[i])
        colors = np.zeros_like(x[i])

        for j in range(num_parts):
            colors[parts[i] == j] = palette[j,:]
        
        point_cloud = o3d.geometry.PointCloud(
                points = o3d.utility.Vector3dVector(x[i]),
            ).translate(centers[i])
        
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)
    
    o3d.visualization.draw_geometries(geometries, window_name=title if title is not None else "Open3D")

def quick_vis_many(parts, title=None):

    geometries = []
    
    for i, part in enumerate(parts):
        geometries.append(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(part.cpu().detach().numpy())
            ).paint_uniform_color(palette[i])
        )

    o3d.visualization.draw_geometries(geometries, 
                                      window_name=title if title is not None else "Open3D")    

def vis_with_vectors(sample, pid, vectors):
    
    pcs = []
    arrs = []
    for i in pid.unique():
        col = palette[i.item()]
        part = sample[pid == i]
        centroid = part.mean(0)
        vector = vectors[i.item()]
        vector = vector / (vector * vector).sum().sqrt()
        arr = create_arrow(vector).paint_uniform_color(col).translate(centroid.cpu().numpy())
        pcs.append(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(part.cpu().detach().numpy())
            ).paint_uniform_color(col)
        )
        arrs.append(arr)
    
    o3d.visualization.draw_geometries(pcs + arrs)

def quick_vis_with_arrows(x, vectors, title=None):

    #generating arrows
    arrows = create_arrows(vectors)
    #calculating the center of each part
    centers = [p.mean(dim=0).cpu().detach().numpy() for p in x]
    
    #translating each arrow to its respective center
    arrows = [arrow.translate(center) for arrow, center in zip(arrows, centers)]

    #transforming to numpy if needed    
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu().numpy()

    if isinstance(x, list):
        x = [p.cpu().numpy() for p in x]

    if isinstance(vectors, list):
        vectors = [v.cpu().numpy() for v in vectors]

    geometries = []

    print(type(x[0]), x[0].shape)

    #keeping a list of the geometries
    for i in range(len(x)):    
        
        point_cloud = o3d.geometry.PointCloud(
                points = o3d.utility.Vector3dVector(x[i])
            ).paint_uniform_color(palette[i % len(palette)])
        
        geometries.append(point_cloud)
        arrows[i] = arrows[i].paint_uniform_color(palette[i % len(palette)])
    
    o3d.visualization.draw_geometries(geometries + arrows, window_name=title if title is not None else "Open3D")

def pc_to_o3d(pc): # point cloud as np.array or torch.tensor
    "turn a point cloud, represented as a np.array or torch.tensor to an [Open3D.geometry.PointCloud](http://www.open3d.org/docs/0.16.0/python_api/open3d.geometry.PointCloud.html)"
    pc = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pc)
    )
    return pc

def quick_vis_batch(batch, grid, x_offset=2.5, y_offset=2.5):
    
    assert len(grid) == 2
    assert batch.shape[0] <= np.prod(grid), "Grid cells should be more than the batch items"
            
    x_offset_start = - x_offset * grid[0] // 2
    x_offset_start = x_offset_start + x_offset / 2 if grid[0] % 2 == 0 else x_offset_start
    
    y_offset_start = - y_offset * grid[1] // 2
    y_offset_start = y_offset_start + y_offset / 2 if grid[1] % 2 == 0 else y_offset_start
    
    pcts = []
    
    k=0
    for i in range(grid[0]):
        for j in range(grid[1]):
            
            try:
                # get point cloud to cpu
                pc = batch[k].detach().cpu()
            except:
                continue

            # translate the point cloud properly
            pc[:, 0] += x_offset_start + i * x_offset
            pc[:, 1] += y_offset_start + j * y_offset
            
            # turn in into an open3d point cloud
            pct = pc_to_o3d(pc)
            
            # append it to the pcts list
            pcts.append(pct)
            
            # incriment k
            k+=1
            
    o3d.visualization.draw_geometries(pcts)

def plot_histogram(data, num_bins, title=""):
    plt.hist(data, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def visualize_distribution_over_time(model_outputs):
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(model_outputs, cmap='magma', cbar_kws={'label': 'Model Outputs'})

    plt.title('Distribution of Model Outputs Across Epochs')
    plt.xlabel('Training Samples')
    plt.ylabel('Epochs')
    plt.show()

def visualize_matrix(matrix):
    """
    Visualize a matrix with brightness corresponding to scalar values.

    Parameters:
    - matrix: 2D numpy array (matrix of scalars)
    """

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().detach().numpy()

    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Matrix Visualization')
    plt.show()

def rotation_matrix_from_vectors2(vec1, vec2):
    
    # Ensure the vectors are of shape (3,)
    assert vec1.shape == (3,)
    assert vec2.shape == (3,)

    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
    
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().detach().numpy()

    # Normalize the vectors
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)    

    if n1 < 1e-7 or n2 < 1e-7:
        return np.eye(3)

    a = vec1 / n1
    b = vec2 / n2

    # Compute the cross product and the sine of the angle
    v = np.cross(a, b)
    s = np.linalg.norm(v)

    # If the vectors are already aligned, return the identity matrix
    if s == 0:
        if np.dot(a, b) > 0:
            return np.eye(3)  # Same dirsection
        else:
            # Vectors are anti-aligned
            # Find a perpendicular vector
            perp = np.array([1, 0, 0]) if not np.allclose(a, np.array([1, 0, 0])) else np.array([0, 1, 0])
            v = np.cross(a, perp)
            v = v / np.linalg.norm(v)
            return -np.eye(3) + 2 * np.outer(v, v)
    
    # Compute the cosine of the angle
    c = np.dot(a, b)

    # Compute the skew-symmetric cross-product matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Compute the rotation matrix using the Rodrigues' rotation formula
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    return R

@torch.no_grad()
def rotation_matrix_from_vectors_batch1(vecs1, vecs2):

    """
    Calculate a batch of rotation matrices that align each vector in v1 with the corresponding vector in v2.
    
    Parameters:
    v1: Tensor of shape (N, 3) - initial vectors
    v2: Tensor of shape (N, 3) - target vectors
    
    Returns:
    rotation_matrices: Tensor of shape (N, 3, 3) - batch of rotation matrices
    """

    #sanity checks    
    assert vecs1.shape == vecs2.shape, "Input tensors must have the same shape"
    assert vecs1.shape[-1] == vecs1.shape[-1] == 3, "Input tensors must have a last dimension of size 3"

    vecs1 = vecs1.double()
    vecs2 = vecs2.double()

    #calculating the norm
    n1 = torch.norm(vecs1, dim=-1, keepdim=True)
    n2 = torch.norm(vecs2, dim=-1, keepdim=True)

    #replacing zero vectors with a fixed vector
    vecs1[n1.squeeze() < 1e-8] == torch.tensor([0, 0, 1], device=vecs1.device).to(vecs1.dtype)
    vecs2[n2.squeeze() < 1e-8] == torch.tensor([0, 0, 1], device=vecs2.device).to(vecs2.dtype)
    n1[n1 < 1e-8] = 1
    n2[n2 < 1e-8] = 1

    #normalizing to unit length
    vecs1 = vecs1 / n1
    vecs2 = vecs2 / n2

    print(torch.cat((vecs1[:20], vecs2[:20]), dim=-1))

    #find the axes of rotation using cross product
    cross_prod = torch.cross(vecs1, vecs2, dim=-1)

    #compute the dot product (cosine of the rotation angle)
    dot_prod = (vecs1 * vecs2).sum(dim=-1, keepdim=True)

    #compute the sin of the angle using the norm of the cross product
    sin_theta = cross_prod.norm(dim=-1, keepdim=True)

    # Handle cases where vectors are nearly parallel or anti-parallel
    parallel_or_antiparallel_mask = sin_theta < 1e-6
    rotation_matrices = torch.eye(3, device=vecs1.device).unsqueeze(0).repeat(vecs1.shape[0], 1, 1).double()
    
    if parallel_or_antiparallel_mask.any():
        non_parallel_idx = (~parallel_or_antiparallel_mask).nonzero(as_tuple=True)[0]
        cross_prod_non_parallel = cross_prod[non_parallel_idx]
        dot_prod_non_parallel = dot_prod[non_parallel_idx]
        sin_theta_non_parallel = sin_theta[non_parallel_idx]

        # Create skew-symmetric cross-product matrix
        K = torch.zeros((cross_prod_non_parallel.shape[0], 3, 3), device=vecs1.device)
        K[:, 0, 1] = -cross_prod_non_parallel[:, 2]
        K[:, 0, 2] = cross_prod_non_parallel[:, 1]
        K[:, 1, 0] = cross_prod_non_parallel[:, 2]
        K[:, 1, 2] = -cross_prod_non_parallel[:, 0]
        K[:, 2, 0] = -cross_prod_non_parallel[:, 1]
        K[:, 2, 1] = cross_prod_non_parallel[:, 0]
        
        # Compute the rotation matrices using the Rodrigues' rotation formula
        I = torch.eye(3, device=vecs1.device).unsqueeze(0)
        K2 = torch.matmul(K, K)
        dot_prod_non_parallel = dot_prod_non_parallel.unsqueeze(-1)  # Shape: (M, 1, 1)
        sin_theta_non_parallel = sin_theta_non_parallel.unsqueeze(-1)  # Shape: (M, 1, 1)
        
        rotation_matrices[non_parallel_idx] = I + K + K2 * ((1 - dot_prod_non_parallel) / (sin_theta_non_parallel ** 2))
    
    return rotation_matrices.float()

@torch.no_grad()
def rotation_matrix_from_vectors_batch(v1, v2, eps=1e-6):
    
    '''
        Calculate a batch of rotation matrices that align each vector in v1 with the corresponding vector in v2.
        
        Parameters:
        v1: Tensor of shape (N, 3) - initial vectors
        v2: Tensor of shape (N, 3) - target vectors
        
        Returns:
        rotation_matrices: Tensor of shape (N, 3, 3) - batch of rotation matrices
    '''

    v1, v2 = v1.double(), v2.double()
    assert v1.shape == v2.shape, "Input tensors must have the same shape"
    assert v1.shape[-1] == v2.shape[-1] == 3, "Input tensors must have a last dimension of size 3"

    N = v1.shape[0]

    # Normalize v1 and v2
    n1, n2 = v1.norm(dim=-1, keepdim=True).clamp(min=eps), v2.norm(dim=-1, keepdim=True).clamp(min=eps)
    v1_norm = v1 / n1
    v2_norm = v2 / n2

    # Compute the dot product (equal to costheta) - (N)
    dot_product = (v1_norm * v2_norm).sum(dim=-1)
    
    # Initialize the rotation matrices as identity matrices - (N, 3, 3)
    R = torch.eye(3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)
    
    # Handle parallel vectors (dot product close to 1)
    parallel_mask = (dot_product >= (1 - eps))
    R[parallel_mask] = torch.eye(3, device=v1.device)
    
    # Handle vectors with extremely low magnitudes
    low_magnitude_mask = (v1.norm(dim=1) < eps) | (v2.norm(dim=1) < eps)
    R[low_magnitude_mask] = torch.eye(3, device=v1.device)

    # Handle anti-parallel vectors (dot product close to -1)
    anti_parallel_mask = (dot_product <= (-1 + eps))
    if anti_parallel_mask.any():

        #find a perpendicular vector to v1 
        v_perpendicular = torch.cross(v1_norm, torch.tensor([1.0, 0.0, 0.0], device=v1.device).double().expand_as(v1_norm))
        #try (1,0,0) and (0,1,0) if v1 is parallel to (0,0,1)
        v_perpendicular[torch.norm(v_perpendicular, dim=1) < eps] = torch.cross(v1_norm[torch.norm(v_perpendicular, dim=1) < eps], torch.tensor([0.0, 1.0, 0.0], device=v1.device).double())
        #normalize it
        v_perpendicular = v_perpendicular / v_perpendicular.norm(dim=1, keepdim=True)
        
        #calculate the skew symmetric matrix K
        K = torch.zeros((v1.size(0), 3, 3), device=v1.device)
        K[:, 0, 1] = -v_perpendicular[:, 2]
        K[:, 0, 2] = v_perpendicular[:, 1]
        K[:, 1, 0] = v_perpendicular[:, 2]
        K[:, 1, 2] = -v_perpendicular[:, 0]
        K[:, 2, 0] = -v_perpendicular[:, 1]
        K[:, 2, 1] = v_perpendicular[:, 0]
        
        #K ^ 2
        K_square = torch.bmm(K, K)
        
        #for 180 degree rotation the rodrigues formula gives R = I + 2K^2
        R[anti_parallel_mask] = torch.eye(3, device=v1.device) + 2 * K_square[anti_parallel_mask]
    
    # Handle the general case
    valid_mask = ~(low_magnitude_mask | parallel_mask | anti_parallel_mask)
    if valid_mask.any():

        # Compute the cross product and its norm
        cross_product = torch.cross(v1_norm, v2_norm, dim=1)
        cross_product_norm = cross_product.norm(dim=1, keepdim=True).clamp(min=eps)
        cross_product_norm = cross_product_norm[valid_mask].unsqueeze(-1)

        # Compute the angle of rotation using the dot product
        angles = torch.acos(dot_product.clamp(-1 + eps, 1 - eps))
        
        # Create a matrix K for each cross product using the skew-symmetric form
        K = torch.zeros((v1.size(0), 3, 3), device=v1.device)
        K[:, 0, 1] = -cross_product[:, 2]
        K[:, 0, 2] = cross_product[:, 1]
        K[:, 1, 0] = cross_product[:, 2]
        K[:, 1, 2] = -cross_product[:, 0]
        K[:, 2, 0] = -cross_product[:, 1]
        K[:, 2, 1] = cross_product[:, 0]
        
        # Compute the rotation matrices using the Rodrigues' rotation formula
        identity = torch.eye(3, device=v1.device).unsqueeze(0).repeat(v1.size(0), 1, 1)[valid_mask]
        K_square = torch.bmm(K, K)[valid_mask]
        K = K[valid_mask]

        sintheta = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)[valid_mask]
        costheta = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)[valid_mask]

        R[valid_mask] = ((identity + sintheta / cross_product_norm * K + (1 - costheta) / (cross_product_norm ** 2) * K_square)).float()
    

    return R

def create_arrows(vectors, heights = None):

    assert vectors.shape[-1] == 3
    if isinstance(heights, list):
        heights = np.array(heights)

    cyl_radius = 0.01
    cone_radius = 0.05
    cone_height = 0.1
    cyl_heights = (heights - cone_height) if heights is not None else [1 - cone_height for _ in vectors] 

    R = [rotation_matrix_from_vectors(np.array([0, 0, 1]), v) for v in vectors]
    arrows = [o3d.geometry.TriangleMesh.create_arrow(cylinder_radius = cyl_radius, cone_radius = cone_radius, cylinder_height = cyl_height, cone_height = cone_height) for cyl_height in cyl_heights]
    arrows = [arrow.rotate(R[i], [0, 0, 0]) for i, arrow in enumerate(arrows)]

    return arrows

def create_arrow(vector):

    assert vector.shape[0] == 3

    vector = vector.cpu().detach()

    height = np.linalg.norm(vector)
    cyl_radius = 0.01
    cone_radius = 0.05
    cone_height = 0.1
    cyl_height = height - cone_height
    if cyl_height < 0:
        cyl_height = 0.1

    R = rotation_matrix_from_vectors2(np.array([0, 0, 1]), vector)
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius = cyl_radius, cone_radius = cone_radius, cylinder_height = cyl_height, cone_height = cone_height)
    arrow = arrow.rotate(R, [0, 0, 0])

    return arrow

if __name__ == "__main__":

    # num_epochs = 10
    # num_samples = 100
    # model_outputs = np.random.rand(num_epochs, num_samples)

    # visualize_distribution_over_time(model_outputs)

    #Testing singular rotation matrix
    v1 = torch.Tensor([0.5, 0.3, 0])
    v2 = torch.Tensor([0, 0, -1])

    R = torch.from_numpy(rotation_matrix_from_vectors2(v1, v2)).float()
    print(R)

    v1 = (R @ v1.unsqueeze(-1)).squeeze()
    print(v1)


    #Testing the rotation matrix batches
    # v1 = torch.Tensor([[1, 0, 0], [0.8, 0.6, 0], [0.5, 0, 0]])
    # v2 = torch.Tensor([[-1, 0, 0], [0, 1, 0],   [0, -0.5, 0]])

    # R = rotation_matrix_from_vectors_batch(v1, v2)

    # print(R)

    # v1 = (R @ v1.unsqueeze(-1)).squeeze()
    # print(v1)
    # print(v2)