import torch
from torch.utils.cpp_extension import load_inline


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int COLS = 32;

template<int TPC, int MAX_M>
__global__ void __launch_bounds__(COLS * TPC)
prlut_kernel(
    const int* __restrict__ activations,
    const int* __restrict__ weights,
    const half* __restrict__ centroids,
    const float* __restrict__ min_max,
    float* __restrict__ partials,
    int M,
    int N,
    int packed_k
) {
    // Entry-major layout: each lane reads a separate shared-memory bank.
    __shared__ float table[256 * COLS];
    const int lane = threadIdx.x % COLS;
    const int sub = threadIdx.x / COLS;
    const int col_start = blockIdx.x * COLS;
    const int col = col_start + lane;
    const int split = blockIdx.y;
    const int split_k = gridDim.y;

    // Partition packed K across blocks, then across threads sharing a column.
    const int chunk = packed_k / split_k;
    const int remainder = packed_k % split_k;
    const int split_start = split * chunk + min(split, remainder);
    const int split_size = chunk + (split < remainder ? 1 : 0);
    const int thread_chunk = split_size / TPC;
    const int thread_remainder = split_size % TPC;
    const int k_start = split_start + sub * thread_chunk + min(sub, thread_remainder);
    const int k_size = thread_chunk + (sub < thread_remainder ? 1 : 0);
    const bool active = col < N;
    const int* weight_ptr = weights + (long long)k_start * N + col;
    const int* activation_ptr = activations + k_start;

    // Start global loads before building the shared product table.
    constexpr int PREFETCH = 4;
    int weight_buffer[PREFETCH];
    int activation_buffer[PREFETCH];
    #pragma unroll
    for (int j = 0; j < PREFETCH; ++j) {
        weight_buffer[j] = (active && j < k_size)
            ? __ldg(weight_ptr + (long long)j * N) : 0;
        activation_buffer[j] = (MAX_M == 1 && j < k_size)
            ? __ldg(activation_ptr + j) : 0;
    }

    // Each thread builds all activation entries for one codebook value.
    for (int p = threadIdx.x; p < 16 * COLS; p += COLS * TPC) {
        const int c = p % COLS;
        const int weight_index = p / COLS;
        const int n = col_start + c;
        const float centroid = (n < N)
            ? __half2float(centroids[(long long)n * 16 + weight_index]) : 0.f;
        #pragma unroll
        for (int activation_index = 0; activation_index < 16; ++activation_index) {
            table[(weight_index * 16 + activation_index) * COLS + c]
                = centroid * (float)activation_index;
        }
    }
    __syncthreads();

    float accumulators[MAX_M];
    float alternate_accumulators[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        accumulators[m] = 0.f;
        alternate_accumulators[m] = 0.f;
    }
    const float* column_table = table + lane;

    if (active) {
        for (int k = 0; k < k_size; k += PREFETCH) {
            int next_weights[PREFETCH];
            int next_activations[PREFETCH];
            #pragma unroll
            for (int j = 0; j < PREFETCH; ++j) {
                const int next_k = k + PREFETCH + j;
                next_weights[j] = (next_k < k_size)
                    ? __ldg(weight_ptr + (long long)next_k * N) : 0;
                next_activations[j] = (MAX_M == 1 && next_k < k_size)
                    ? __ldg(activation_ptr + next_k) : 0;
            }

            #pragma unroll
            for (int j = 0; j < PREFETCH; ++j) {
                if (k + j < k_size) {
                    const int weight_word = weight_buffer[j];
                    #pragma unroll
                    for (int m = 0; m < MAX_M; ++m) {
                        if (m < M) {
                            const int activation_word = (MAX_M == 1)
                                ? activation_buffer[j]
                                : __ldg(activation_ptr + (long long)m * packed_k + k + j);
                            float sum = 0.f;
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int weight_index = (weight_word >> (4 * i)) & 15;
                                const int activation_index = (activation_word >> (4 * i)) & 15;
                                const int entry = (weight_index << 4) | activation_index;
                                sum += column_table[entry * COLS];
                            }
                            if (j & 1) {
                                alternate_accumulators[m] += sum;
                            } else {
                                accumulators[m] += sum;
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < PREFETCH; ++j) {
                weight_buffer[j] = next_weights[j];
                activation_buffer[j] = next_activations[j];
            }
        }
    }

    // Reuse the product table for the threads-per-column reduction.
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        table[(sub * MAX_M + m) * COLS + lane]
            = accumulators[m] + alternate_accumulators[m];
    }
    __syncthreads();

    if (sub == 0 && col < N) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sum = 0.f;
                #pragma unroll
                for (int t = 0; t < TPC; ++t) {
                    sum += table[(t * MAX_M + m) * COLS + lane];
                }
                const float minimum = min_max[2 * m];
                const float scale = (min_max[2 * m + 1] - minimum) / 15.f;
                partials[((long long)split * M + m) * N + col] = scale * sum;
            }
        }
    }
}

__global__ void prlut_reduce(
    const float* __restrict__ partials,
    const float* __restrict__ min_max,
    const float* __restrict__ weight_sums,
    half* __restrict__ output,
    int split_k,
    int M,
    int N
) {
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) {
        return;
    }

    float sum = min_max[2 * m] * weight_sums[n];
    for (int split = 0; split < split_k; ++split) {
        sum += partials[((long long)split * M + m) * N + n];
    }
    output[(long long)m * N + n] = __float2half(sum);
}

void prlut(
    torch::Tensor activations,
    torch::Tensor weights,
    torch::Tensor centroids,
    torch::Tensor min_max,
    torch::Tensor weight_sums,
    torch::Tensor partials,
    torch::Tensor output,
    int M,
    int N,
    int packed_k,
    int split_k,
    int threads_per_column
) {
    TORCH_CHECK(M >= 1 && M <= 16, "Decoding supports 1 <= M <= 16");
    TORCH_CHECK(N > 0 && packed_k > 0 && split_k > 0,
                "N, packed_k and split_k must be positive");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((N + COLS - 1) / COLS, split_k);
    const int max_m = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16;

    #define LAUNCH(T, ROWS) \
        prlut_kernel<T, ROWS><<<grid, COLS * T, 0, stream>>>( \
            (const int*)activations.data_ptr(), (const int*)weights.data_ptr(), \
            (const half*)centroids.data_ptr(), (const float*)min_max.data_ptr(), \
            (float*)partials.data_ptr(), M, N, packed_k)

    #define DISPATCH_ROWS(T) \
        switch (max_m) { \
            case 1: LAUNCH(T, 1); break; \
            case 2: LAUNCH(T, 2); break; \
            case 4: LAUNCH(T, 4); break; \
            case 8: LAUNCH(T, 8); break; \
            default: LAUNCH(T, 16); break; \
        }

    switch (threads_per_column) {
        case 4: DISPATCH_ROWS(4); break;
        case 8: DISPATCH_ROWS(8); break;
        case 16: DISPATCH_ROWS(16); break;
        default: TORCH_CHECK(false, "threads_per_column must be 4, 8 or 16");
    }
    #undef LAUNCH
    #undef DISPATCH_ROWS

    prlut_reduce<<<dim3((N + 255) / 256, M), 256, 0, stream>>>(
        (const float*)partials.data_ptr(), (const float*)min_max.data_ptr(),
        (const float*)weight_sums.data_ptr(), (half*)output.data_ptr(), split_k, M, N
    );
}
"""

_CPP_DECLARATIONS = """
void prlut(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    int, int, int, int, int
);
"""

_compute_capability = torch.cuda.get_device_capability()
pr = load_inline(
    name="clusquant_lut",
    cpp_sources=_CPP_DECLARATIONS,
    cuda_sources=_CUDA_SOURCE,
    functions=["prlut"],
    with_cuda=True,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        f"-arch=sm_{_compute_capability[0]}{_compute_capability[1]}",
    ],
)


def unpack_idx(packed_weights: torch.Tensor,
               in_features: int) -> torch.Tensor:
    """Unpack int32 weight indices [K/8, N] into int64 indices [K, N]."""
    packed_k, out_features = packed_weights.shape
    indices = torch.empty(
        packed_k * 8,
        out_features,
        dtype=torch.long,
        device=packed_weights.device,
    )
    for i in range(8):
        indices[i::8] = (packed_weights >> (4 * i)) & 15
    return indices[:in_features]


def precompute_csum(packed_weights: torch.Tensor,
                    centroids: torch.Tensor,
                    in_features: int) -> torch.Tensor:
    """Sum reconstructed weights over K, producing FP32 weight sums [N].

    This is weight-only preprocessing performed once at model load time.
    Codebooks are [N, 16], and packed weight indices are [K/8, N].
    """
    indices = unpack_idx(packed_weights, in_features)
    out_features = centroids.shape[0]
    channels = torch.arange(out_features, device=centroids.device)
    channels = channels.unsqueeze(0).expand(in_features, out_features)
    return centroids.float()[channels, indices].sum(dim=0).contiguous()
