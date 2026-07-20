#include "gds_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

// O_DIRECT is Linux-specific (GDS itself is Linux-only); on other platforms
// (e.g. macOS dev machines) fall back to buffered open so the file still
// compiles and the stub path remains testable.
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

// cuFile is optional: when MINIFLEX_WITH_CUFILE is defined at build time and
// the headers are present we compile the real GDS path; otherwise the class
// is a stub that reports is_available()==false so the Python worker falls
// back to the CPU two-hop path.
#ifdef MINIFLEX_WITH_CUFILE
#include <cufile.h>
#endif

namespace miniflex {

GDSIOCTX::GDSIOCTX(int64_t blocks_per_file,
                   std::vector<char*> gpu_ptrs,
                   int64_t layer_num,
                   int64_t kv_dim,
                   int64_t gpu_num_blocks,
                   int64_t slice_bytes,
                   int64_t gpu_block_step,
                   int64_t gpu_kv_pitch,
                   std::vector<std::string> file_paths)
    : blocks_per_file_(blocks_per_file),
      gpu_ptrs_(std::move(gpu_ptrs)),
      layer_num_(layer_num),
      kv_dim_(kv_dim),
      gpu_num_blocks_(gpu_num_blocks),
      slice_bytes_(slice_bytes),
      gpu_block_step_(gpu_block_step),
      gpu_kv_pitch_(gpu_kv_pitch),
      file_paths_(std::move(file_paths)) {
  init();
}

GDSIOCTX::~GDSIOCTX() {
#ifdef MINIFLEX_WITH_CUFILE
  if (available_) {
    for (char* p : gpu_ptrs_) {
      cuFileBufDeregister(p);
    }
    cuFileDriverClose();
  }
#endif
  for (int fd : fds_) {
    if (fd >= 0) {
      close(fd);
    }
  }
}

void GDSIOCTX::init() {
  for (const auto& path : file_paths_) {
    // O_DIRECT is required for real GDS; without it cuFile silently falls
    // back to an internal bounce buffer, which defeats the purpose.
    int fd = open(path.c_str(), O_RDWR | O_DIRECT);
    if (fd < 0) {
      // Fall back to buffered open so non-GDS filesystems still work via
      // cuFile's internal compatibility path.
      fd = open(path.c_str(), O_RDWR);
    }
    if (fd < 0) {
      fprintf(stderr, "[GDSIOCTX] open failed for %s: %s\n", path.c_str(),
              strerror(errno));
      return;
    }
    fds_.push_back(fd);
  }
  if (fds_.size() != file_paths_.size()) {
    return;
  }

#ifdef MINIFLEX_WITH_CUFILE
  CUfileError_t st = cuFileDriverOpen();
  if (st.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "[GDSIOCTX] cuFileDriverOpen failed: %d\n", st.err);
    return;
  }
  for (char* p : gpu_ptrs_) {
    // Register the whole GPU tensor region; size = num_blocks * block stride.
    size_t sz = static_cast<size_t>(gpu_num_blocks_) *
                static_cast<size_t>(gpu_block_step_);
    CUfileError_t rst = cuFileBufRegister(p, sz, 0);
    if (rst.err != CU_FILE_SUCCESS) {
      fprintf(stderr, "[GDSIOCTX] cuFileBufRegister failed: %d\n", rst.err);
      // Continue registering the rest; transfer will report failure per-op.
    }
  }
  available_ = true;
#else
  fprintf(stderr,
          "[GDSIOCTX] built without MINIFLEX_WITH_CUFILE; GDS path disabled, "
          "falling back to CPU two-hop.\n");
  available_ = false;
#endif
}

auto GDSIOCTX::transfer_blocks(const torch::Tensor& src_block_ids,
                               const torch::Tensor& dst_block_ids,
                               bool is_read) -> bool {
  if (!available_) {
    return false;
  }
  // src/dst ordering matches the rest of MiniFlex: for reads (DISK2D) src is
  // the SSD side, dst is GPU; for writes (D2DISK) src is GPU, dst is SSD.
  if (is_read) {
    return do_transfer(dst_block_ids, src_block_ids, /*is_read=*/true);
  }
  return do_transfer(src_block_ids, dst_block_ids, /*is_read=*/false);
}

auto GDSIOCTX::do_transfer(const torch::Tensor& gpu_block_ids,
                           const torch::Tensor& ssd_block_ids,
                           bool is_read) -> bool {
#ifdef MINIFLEX_WITH_CUFILE
  TORCH_CHECK(gpu_block_ids.numel() == ssd_block_ids.numel(),
              "gpu/ssd block id count mismatch");
  const int64_t n = gpu_block_ids.numel();
  if (n == 0) {
    return true;
  }
  const int64_t* gpu_ids = gpu_block_ids.data_ptr<int64_t>();
  const int64_t* ssd_ids = ssd_block_ids.data_ptr<int64_t>();
  const int64_t block_bytes = slice_bytes_ * layer_num_ * kv_dim_;

  for (int64_t i = 0; i < n; ++i) {
    const int64_t g = gpu_ids[i];
    const int64_t s = ssd_ids[i];
    if (g < 0 || g >= gpu_num_blocks_ || s < 0) {
      fprintf(stderr, "[GDSIOCTX] block id out of range: gpu=%ld ssd=%ld\n", g, s);
      return false;
    }
    const int64_t file_idx = s / blocks_per_file_;
    const int64_t block_idx = s % blocks_per_file_;
    if (file_idx < 0 || file_idx >= static_cast<int64_t>(fds_.size())) {
      fprintf(stderr, "[GDSIOCTX] file index out of range: %ld\n", file_idx);
      return false;
    }
    const int fd = fds_[file_idx];
    const off_t file_off = block_idx * block_bytes;

    // Each block is stored as (layer, k/v) chunks of slice_bytes, laid out
    // contiguously on disk; on the GPU side the same chunk lives at
    // layer_ptr + block * block_step + kv * kv_pitch.
    off_t chunk_off = file_off;
    for (int64_t layer = 0; layer < layer_num_; ++layer) {
      char* gpu_base = gpu_ptrs_[layer] + g * gpu_block_step_;
      for (int64_t kv = 0; kv < kv_dim_; ++kv) {
        void* dev_ptr = gpu_base + kv * gpu_kv_pitch_;
        ssize_t ret;
        if (is_read) {
          ret = cuFileRead(fd, dev_ptr, slice_bytes_, chunk_off, 0);
        } else {
          ret = cuFileWrite(fd, dev_ptr, slice_bytes_, chunk_off, 0);
        }
        if (ret < 0 || ret != slice_bytes_) {
          fprintf(stderr,
                  "[GDSIOCTX] %s failed: block gpu=%ld ssd=%ld layer=%ld kv=%ld "
                  "ret=%zd expected=%ld\n",
                  is_read ? "cuFileRead" : "cuFileWrite", g, s, layer, kv, ret,
                  slice_bytes_);
          return false;
        }
        chunk_off += slice_bytes_;
      }
    }
  }
  return true;
#else
  (void)gpu_block_ids;
  (void)ssd_block_ids;
  (void)is_read;
  return false;
#endif
}

}  // namespace miniflex
