#ifndef MINIFLEX_GDS_IO_H
#define MINIFLEX_GDS_IO_H

// GPU Direct Storage (GDS) transfer context.
//
// Moves KV cache blocks directly between SSD files and GPU memory via the
// cuFile API, bypassing the CPU bounce buffer entirely.  This is the fast
// path for SSD<->GPU; the existing SSDCPUTransferWorker (io_uring to a CPU
// tensor) remains as the portable fallback.
//
// Layout contract (same as the rest of MiniFlex):
//   - GPU tensors are per-layer, LAYERFIRST or LAYERBLOCK; the worker passes
//     per-layer base pointers plus byte strides, this class does not know
//     the logical layout.
//   - SSD files store blocks BLOCKFIRST: block b lives at file offset
//     b * block_bytes, sliced per (layer, k/v) chunk of slice_bytes.
//
// GDS requires:
//   - an NVIDIA GPU with GDS support + the cuFile library (libcufile.so),
//   - GPU memory registered with cuFileBufRegister,
//   - files on a GDS-capable filesystem (or cuFile falls back internally).
// When cuFile is unavailable at build time the class still compiles (stubbed
// out) and reports is_available() == false so the worker can fall back to the
// CPU two-hop path.

#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <vector>

namespace miniflex {

class GDSIOCTX {
 public:
  GDSIOCTX(int64_t blocks_per_file,
           std::vector<char*> gpu_ptrs,
           int64_t layer_num,
           int64_t kv_dim,
           int64_t gpu_num_blocks,
           int64_t slice_bytes,
           int64_t gpu_block_step,
           int64_t gpu_kv_pitch,
           std::vector<std::string> file_paths);
  ~GDSIOCTX();

  // is_read=true  -> SSD -> GPU (DISK2D)
  // is_read=false -> GPU -> SSD (D2DISK)
  auto transfer_blocks(const torch::Tensor& src_block_ids,
                       const torch::Tensor& dst_block_ids,
                       bool is_read) -> bool;

  // True when cuFile was found at build time and initialized successfully.
  auto is_available() const -> bool { return available_; }

 private:
  void init();
  auto do_transfer(const torch::Tensor& gpu_block_ids,
                   const torch::Tensor& ssd_block_ids,
                   bool is_read) -> bool;

  int64_t blocks_per_file_;
  std::vector<char*> gpu_ptrs_;
  int64_t layer_num_;
  int64_t kv_dim_;
  int64_t gpu_num_blocks_;
  int64_t slice_bytes_;
  int64_t gpu_block_step_;
  int64_t gpu_kv_pitch_;
  std::vector<std::string> file_paths_;
  std::vector<int> fds_;
  bool available_ = false;
};

}  // namespace miniflex

#endif  // MINIFLEX_GDS_IO_H
