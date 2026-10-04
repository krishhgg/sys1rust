# MLX headers

These 5 Metal headers are verbatim copies from MLX 0.32.2 (https://github.com/ml-explore/mlx).
They were copied byte for byte from the `mlx` 0.32.2 Python wheel, which installs them under
`mlx/include/mlx/backend/metal/kernels/steel/`. In the MLX repository they are under
`mlx/backend/metal/kernels/steel/`:

- `steel/defines.h`
- `steel/utils/type_traits.h`
- `steel/utils/integral_constant.h`
- `steel/gemm/nax.h`
- `steel/gemm/gemm_nax.h`

No file here is edited. `../../nax_gemm.rs` builds the header of the `nax` kernels from them
at load. It leaves out each `#pragma once` line and each `#include "mlx/..."` line, because
MLX compiles a custom kernel from one string with no include path. The test
`mlx_headers_are_verbatim` compares each file with the MLX build the runtime links when
`MLX_SYS_PREBUILT_DIR` is set.

MLX is Copyright © 2023 Apple Inc. and released under the MIT License. `LICENSE` here is
MLX's license file, unchanged.
