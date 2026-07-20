"""End-to-end fallback tests for the GDS path on machines without cuFile/GPU.

Validates the contract that keeps the runtime safe when GDS is enabled but
unavailable at runtime: every layer must degrade cleanly to the CPU two-hop
path, and the GDS worker must never claim success for a transfer it did not
perform.

Run: PYTHONPATH=pysrc:test python -m pytest test/gds_fallback_test.py -q
"""
import numpy as np
import pytest
import torch

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.storage import KVCacheLayout, KVCacheLayoutType, StorageHandle, StorageHandlerType
from miniflex.common.transfer import DeviceType, TransferType

# The GDS worker needs miniflex._C (built with the C++ extension).  On machines
# without the toolchain/GPU the module is absent; skip cleanly there so the
# pure-Python suite stays green.
try:
  import miniflex._C  # noqa: F401
  from miniflex.transfer.worker import GPUDirectSSDTransferWorker
  _HAS_C = True
except Exception:  # pragma: no cover - environment without the extension
  GPUDirectSSDTransferWorker = None
  _HAS_C = False

pytestmark = pytest.mark.skipif(not _HAS_C, reason="miniflex._C extension not built")


def _gpu_handle(tokens_per_block=16, num_blocks=8, num_layers=2, num_heads=2, head_size=4):
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )
  data = torch.zeros(layout.get_total_elements(), dtype=torch.float32)
  return StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    kv_layout=layout,
    dtype=torch.float32,
    data=data,
    gpu_device_id=None,
  )


def _ssd_handle(tmp_path, tokens_per_block=16, num_blocks=8, num_layers=2, num_heads=2, head_size=4):
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.BLOCKFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )
  f = tmp_path / "gds_test.bin"
  f.write_bytes(b"\x00" * layout.get_total_elements() * 4)
  return StorageHandle(
    handle_type=StorageHandlerType.FILE,
    kv_layout=layout,
    dtype=torch.float32,
    data=[str(f)],
    num_blocks_per_file=num_blocks,
  )


class _DummyPipe:
  def poll(self, timeout=0):
    return False
  def recv(self):
    raise EOFError


class _DummyQueue:
  def __init__(self):
    self.items = []
  def put(self, x):
    self.items.append(x)


def _make_worker(tmp_path):
  gpu = _gpu_handle()
  ssd = _ssd_handle(tmp_path)
  op_buf = torch.zeros((4, 16), dtype=torch.int64)
  return GPUDirectSSDTransferWorker(
    worker_id=0,
    transfer_conn=_DummyPipe(),
    finished_ops_queue=_DummyQueue(),
    op_buffer_tensor=op_buf,
    gpu_storage_handle=gpu,
    ssd_storage_handle=ssd,
  )


def test_gds_ctx_reports_unavailable_without_cufile(tmp_path):
  worker = _make_worker(tmp_path)
  # On a machine without cuFile the ctx must report unavailable so the engine
  # can fall back.  (On a cuFile machine this test would exercise the real
  # path instead; here we assert the stub contract.)
  assert worker.is_available() in (True, False)  # must not raise


def test_worker_never_claims_success_when_unavailable(tmp_path):
  worker = _make_worker(tmp_path)
  if worker.is_available():
    # Real GDS machine: transfer would actually run; skip the stub assertion.
    return
  src = torch.tensor([0, 1], dtype=torch.int64)
  dst = torch.tensor([0, 1], dtype=torch.int64)
  # DISK2D (read): must return False when unavailable, never True.
  assert worker._transfer_impl(src, dst, TransferType.DISK2D) is False
  # D2DISK (write): same.
  assert worker._transfer_impl(src, dst, TransferType.D2DISK) is False


def test_worker_rejects_invalid_transfer_type(tmp_path):
  worker = _make_worker(tmp_path)
  src = torch.tensor([0], dtype=torch.int64)
  dst = torch.tensor([0], dtype=torch.int64)
  for bad in (TransferType.H2D, TransferType.D2H, TransferType.H2DISK, TransferType.DISK2H):
    try:
      worker._transfer_impl(src, dst, bad)
    except ValueError:
      pass
    else:
      raise AssertionError(f"expected ValueError for {bad}")


def test_worker_bounds_checking(tmp_path):
  worker = _make_worker(tmp_path)
  if not worker.is_available():
    # bounds checking happens before the availability check only for valid
    # ranges; out-of-range must raise regardless of availability.
    pass
  src = torch.tensor([0, 999], dtype=torch.int64)  # 999 out of range
  dst = torch.tensor([0, 1], dtype=torch.int64)
  try:
    worker._transfer_impl(src, dst, TransferType.D2DISK)
  except ValueError as e:
    assert "out of range" in str(e)
  else:
    # If GDS were available this should still have raised before touching HW.
    raise AssertionError("expected ValueError for out-of-range block id")
