# coding=utf-8
import numpy as np


def pca_normal(points):
    """Compute surface normal from Nx3 3D points using PCA.

    Returns (normal, centroid). Normal points toward the camera (origin).
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    _, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]  # smallest eigenvalue
    if np.dot(normal, centroid) > 0:
        normal = -normal
    return normal, centroid


def cluster_normals(items, cosine_threshold=0.6):
    """Greedy clustering of (normal, centroid) pairs by cosine similarity.

    Returns list of dicts: [{'normal': ndarray, 'center': ndarray, 'size': int}, ...]
    sorted by cluster size descending.
    """
    if len(items) < 2:
        return [{'normal': items[0][0], 'center': items[0][1], 'size': 1}] if items else []

    normals = np.array([item[0] for item in items])
    n = len(normals)

    # Cosine similarity (absolute value for bidirectional faces)
    cos_sim = np.abs(normals @ normals.T)

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster_idx = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and cos_sim[i, j] > cosine_threshold:
                cluster_idx.append(j)
                visited[j] = True
        clusters.append(cluster_idx)

    clusters.sort(key=len, reverse=True)

    result = []
    for cluster in clusters:
        cluster_normals = normals[cluster]
        avg_normal = np.mean(cluster_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        cluster_centers = np.array([items[j][1] for j in cluster])
        center = np.mean(cluster_centers, axis=0)

        result.append({
            'normal': avg_normal,
            'center': center,
            'size': len(cluster),
        })

    return result


def classify_faces(clusters):
    """Label face clusters as 'top' or 'front' by comparing to reference directions.

    Reference directions (camera frame):
      top face normal:   (0, -1, 0)
      front face normal: (0, 0, 1)

    Returns dict: {'top': {...}, 'front': {...}} or None.
    """
    ref_top = np.array([0, -1, 0])
    ref_front = np.array([0, 0, 1])

    face_groups = {}
    for cluster in clusters[:2]:
        n = cluster['normal']
        sim_top = np.abs(np.dot(n, ref_top))
        sim_front = np.abs(np.dot(n, ref_front))
        label = 'top' if sim_top > sim_front else 'front'
        face_groups[label] = cluster

    return face_groups if face_groups else None


def build_orthonormal_basis(n_top, n_front):
    """Create right-handed orthonormal basis from two approximate face normals.

    Returns (x_axis, y_axis, z_axis) where:
      z_axis = n_top normalized
      x_axis = n_front orthonormalized
      y_axis = cross(z, x)
    """
    z_axis = n_top / np.linalg.norm(n_top)
    x_axis = n_front / np.linalg.norm(n_front)

    # Remove projection of x onto z
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1, 0, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return x_axis, y_axis, z_axis


def rotation_matrix_to_euler(R):
    """Convert 3x3 rotation matrix to ZYX Euler angles (roll, pitch, yaw) in radians."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])


def build_rotation_from_normal(normal, primary_axis):
    """Build orthonormal rotation matrix given one primary axis direction.

    正方体约束：给定一个面的法线，构建完整旋转矩阵。
    - primary_axis='z': normal 是顶面法线 (物体系 z 轴)
    - primary_axis='x': normal 是前面法线 (物体系 x 轴)

    R = [x_obj_in_cam | y_obj_in_cam | z_obj_in_cam]，列向量是物体系三轴在相机系的表示。
    """
    v1 = normal / np.linalg.norm(normal)

    # 选一个不平行于 v1 的参考方向来构造正交基
    if abs(np.dot(v1, [0, 0, 1])) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])

    v2 = np.cross(v1, ref)
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)

    if primary_axis == 'z':    # 顶面: z=v1(normal)
        R = np.column_stack([v2, v3, v1])
    elif primary_axis == 'x':  # 前面: x=v1(normal)
        R = np.column_stack([v1, v2, v3])
    else:
        raise ValueError(f"Unknown primary_axis: {primary_axis}")

    return R


def cube_center_from_face(face_center, face_normal, half_side=175.0):
    """从面中心推算正方体中心: cube_center = face_center - normal * half_side."""
    return face_center - face_normal * half_side
