import torch


def _argmin_dist(matrix: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """argmin_k (matrix - centroids[:, k])^2, using O(groups*samples) memory."""
    group_count, sample_count = matrix.shape
    min_distance = torch.full((group_count, sample_count), float("inf"), dtype=matrix.dtype, device=matrix.device)
    assignments = torch.zeros(group_count, sample_count, dtype=torch.uint8, device=matrix.device)
    for centroid_id in range(centroids.shape[1]):
        distance = (matrix - centroids[:, centroid_id:centroid_id + 1]).pow(2)
        better = distance < min_distance
        min_distance = torch.where(better, distance, min_distance)
        assignments[better] = centroid_id
    return assignments


def _min_dist(matrix: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """min_k (matrix - centroids[:, k])^2, using O(groups*samples) memory."""
    min_distance = torch.full(matrix.shape, float("inf"), dtype=matrix.dtype, device=matrix.device)
    for centroid_id in range(centroids.shape[1]):
        distance = (matrix - centroids[:, centroid_id:centroid_id + 1]).pow(2)
        min_distance = torch.minimum(min_distance, distance)
    return min_distance


def kmeans(matrix: torch.Tensor,
           cluster_num: int,
           kmeans_iteration: int = 300,
           tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster every row of ``matrix`` independently with 1-D k-means.

    Centroids are seeded with k-means++ and refined with Lloyd iterations.

    Args:
        matrix: Values of shape (groups, samples); each row is one clustering problem.
        cluster_num: Number of centroids per row.
        kmeans_iteration: Maximum number of Lloyd iterations.
        tol: Stop early once the largest centroid move falls below this value.

    Returns:
        Tuple of centroids with shape (groups, cluster_num) in ``matrix.dtype``
        and uint8 assignments with shape (groups, samples).
    """
    group_count, sample_count = matrix.shape
    device = matrix.device
    orig_dtype = matrix.dtype

    dtype = torch.float32
    matrix_f32 = matrix.to(dtype)

    group_ids = torch.arange(group_count, device=device).unsqueeze(1).expand(-1, sample_count).reshape(-1)

    if cluster_num == 1:
        centroids = matrix_f32.mean(dim=1, keepdim=True)
    else:
        # k-means++ initialization
        sample_ids = torch.randint(sample_count, (group_count,), device=device)
        centroids = matrix_f32[torch.arange(group_count, device=device), sample_ids].unsqueeze(1)
        for _ in range(1, cluster_num):
            distance = _min_dist(matrix_f32, centroids)
            sample_probs = distance / (distance.sum(dim=1, keepdim=True) + 1e-8)
            sample_ids = torch.multinomial(sample_probs, 1).squeeze(1)
            new_centroid = matrix_f32[torch.arange(group_count, device=device), sample_ids].unsqueeze(1)
            centroids = torch.cat([centroids, new_centroid], dim=1)

        for _ in range(kmeans_iteration):
            assignments = _argmin_dist(matrix_f32, centroids)

            flat_values = matrix_f32.reshape(-1)
            flat_assignments = assignments.reshape(-1)
            flat_cluster_ids = group_ids * cluster_num + flat_assignments

            centroid_sums_1d = torch.zeros(group_count * cluster_num, dtype=dtype, device=device)
            centroid_sums_1d.scatter_add_(0, flat_cluster_ids, flat_values)
            centroid_sums = centroid_sums_1d.view(group_count, cluster_num)

            centroid_counts_1d = torch.zeros(group_count * cluster_num, dtype=dtype, device=device)
            ones = torch.ones_like(flat_values, dtype=dtype)
            centroid_counts_1d.scatter_add_(0, flat_cluster_ids, ones)
            centroid_counts = centroid_counts_1d.view(group_count, cluster_num)

            empty_clusters = (centroid_counts == 0)
            denominator = torch.where(empty_clusters, torch.ones_like(centroid_counts), centroid_counts)
            updated_centroids = centroid_sums / denominator
            new_centroids = torch.where(empty_clusters, centroids, updated_centroids)

            converged = (centroids - new_centroids).abs().max() < tol
            centroids = new_centroids
            del assignments, flat_values, flat_assignments, flat_cluster_ids
            del centroid_sums_1d, centroid_sums, centroid_counts_1d, centroid_counts
            del ones, empty_clusters, denominator, updated_centroids, new_centroids
            if converged:
                break

    final_assign = _argmin_dist(matrix_f32, centroids)
    result = centroids.to(dtype=orig_dtype)
    del matrix_f32, group_ids, centroids
    torch.cuda.empty_cache()
    return result, final_assign
