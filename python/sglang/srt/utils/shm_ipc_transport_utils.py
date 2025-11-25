import fcntl
import logging
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import numpy as np

from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)

# Default shared memory cache size for multimodal features
SHM_IPC_FEATURE_CACHE_SIZE = (
    2 * 1024 * 1024 * 1024
    if not get_int_env_var("SGLANG_SHM_IPC_FEATURE_CACHE_MB")
    else get_int_env_var("SGLANG_SHM_IPC_FEATURE_CACHE_MB") * 1024 * 1024
)


class ShmIPCLock:
    """File-based lock for shared memory IPC operations using fcntl."""

    def __init__(self, lock_file_path: Optional[str] = None):
        """
        Initialize the lock.

        Args:
            lock_file_path: Path to the lock file. If None, uses a default path in /tmp.
        """
        if lock_file_path is None:
            lock_file_path = "/tmp/shm_ipc.lock"
        self.lock_file_path = lock_file_path
        self._lock_file: Optional[object] = None

    def _ensure_lock_file(self):
        """Ensure the lock file exists."""
        if self._lock_file is None:
            # Create the lock file if it doesn't exist
            Path(self.lock_file_path).touch(exist_ok=True)
            self._lock_file = open(self.lock_file_path, "r+")

    def acquire(self, exclusive: bool = True, blocking: bool = True, timeout: Optional[float] = None):
        """
        Acquire the lock.

        Args:
            exclusive: If True, acquire exclusive lock (for writing). If False, acquire shared lock (for reading).
            blocking: If True, block until the lock is acquired. If False, return immediately.
            timeout: Maximum time to wait for the lock (in seconds). Only used if blocking=True.

        Returns:
            True if the lock was acquired, False otherwise.

        Raises:
            IOError: If the lock cannot be acquired.
        """
        self._ensure_lock_file()
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH

        if blocking:
            if timeout is not None:
                # Use non-blocking mode with timeout
                import time

                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(self._lock_file, lock_type | fcntl.LOCK_NB)
                        return True
                    except IOError:
                        time.sleep(0.01)  # Small sleep to avoid busy waiting
                return False
            else:
                # Block indefinitely
                fcntl.flock(self._lock_file, lock_type)
                return True
        else:
            # Non-blocking
            try:
                fcntl.flock(self._lock_file, lock_type | fcntl.LOCK_NB)
                return True
            except IOError:
                return False

    def release(self):
        """Release the lock."""
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            except IOError:
                logger.info(f"Error: Failed to release lock: {self.lock_file_path}")

    def __enter__(self):
        """Context manager entry."""
        self.acquire(exclusive=True)  # Default to exclusive for backward compatibility
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def close(self):
        """Close the lock file."""
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class ShmIPC:
    """Shared memory IPC with locking support for /dev/shm."""

    def __init__(
        self,
        name: str,
        size: int,
        create: bool = True,
        lock_file_path: Optional[str] = None,
    ):
        """
        Initialize shared memory IPC.

        Args:
            name: Name of the shared memory segment.
            size: Size of the shared memory segment in bytes.
            create: If True, create a new shared memory segment. If False, attach to existing.
            lock_file_path: Path to the lock file. If None, uses a default path.
        """
        self.name = name
        self.size = size
        self.create = create
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.lock = ShmIPCLock(lock_file_path)

        if create:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=True, size=size
                )
                # Initialize memory to zero
                self.shm.buf[:] = b"\x00" * size
            except FileExistsError:
                # If already exists, try to attach
                logger.info(
                    f"Error: Shared memory '{name}' already exists, attaching instead"
                )
                self.shm = shared_memory.SharedMemory(name=name)
                if self.shm.size < size:
                    raise ValueError(
                        f"Existing shared memory size {self.shm.size} < required size {size}"
                    )
        else:
            try:
                self.shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Shared memory segment '{name}' not found. Create it first with create=True."
                )
            if self.shm.size < size:
                raise ValueError(
                    f"Shared memory size {self.shm.size} < required size {size}"
                )

    @contextmanager
    def locked_access(self, exclusive: bool = True):
        """
        Context manager for locked access to shared memory.

        Args:
            exclusive: If True, use exclusive lock (for writing). If False, use shared lock (for reading).

        Example:
            # For writing (exclusive lock)
            with shm_ipc.locked_access(exclusive=True):
                shm_ipc.shm.buf[0:10] = b"hello"

            # For reading (shared lock, allows concurrent readers)
            with shm_ipc.locked_access(exclusive=False):
                data = shm_ipc.shm.buf[0:10]
        """
        self.lock.acquire(exclusive=exclusive)
        try:
            yield self.shm.buf
        finally:
            self.lock.release()

    def read(self, offset: int = 0, size: Optional[int] = None) -> bytes:
        """
        Read data from shared memory with lock protection.

        Args:
            offset: Offset in bytes from the start of shared memory.
            size: Number of bytes to read. If None, reads to the end.

        Returns:
            Bytes read from shared memory.
        """
        if size is None:
            size = self.shm.size - offset

        if offset + size > self.shm.size:
            raise ValueError(
                f"Read range [{offset}, {offset + size}) exceeds shared memory size {self.shm.size}"
            )

        with self.locked_access():
            return bytes(self.shm.buf[offset: offset + size])

    def write(self, data: bytes, offset: int = 0):
        """
        Write data to shared memory with lock protection.

        Args:
            data: Data to write.
            offset: Offset in bytes from the start of shared memory.

        Raises:
            ValueError: If the write would exceed shared memory bounds.
        """
        if offset + len(data) > self.shm.size:
            raise ValueError(
                f"Write range [{offset}, {offset + len(data)}) exceeds shared memory size {self.shm.size}"
            )

        with self.locked_access():
            self.shm.buf[offset: offset + len(data)] = data

    def read_numpy(
        self,
        dtype: Union[str, np.dtype],
        shape: tuple,
        offset: int = 0,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Read numpy array from shared memory with lock protection.

        Args:
            dtype: NumPy dtype of the array.
            shape: Shape of the array.
            offset: Offset in bytes from the start of shared memory.
            copy: If True, return a copy of the data. If False, return a view (faster but less safe).

        Returns:
            NumPy array (copy or view) of the shared memory.
        """
        dtype = np.dtype(dtype)
        element_size = dtype.itemsize
        total_size = int(np.prod(shape)) * element_size

        if offset + total_size > self.shm.size:
            raise ValueError(
                f"Read range [{offset}, {offset + total_size}) exceeds shared memory size {self.shm.size}"
            )

        with self.locked_access(exclusive=False):
            view = np.ndarray(
                shape, dtype=dtype, buffer=self.shm.buf, offset=offset
            )
            if copy:
                return view.copy()
            return view

    def write_numpy(self, array: np.ndarray, offset: int = 0):
        """
        Write numpy array to shared memory with lock protection.

        Args:
            array: NumPy array to write.
            offset: Offset in bytes from the start of shared memory.
        """
        total_size = array.nbytes

        if offset + total_size > self.shm.size:
            raise ValueError(
                f"Write range [{offset}, {offset + total_size}) exceeds shared memory size {self.shm.size}"
            )

        with self.locked_access(exclusive=True):
            view = np.ndarray(
                array.shape, dtype=array.dtype, buffer=self.shm.buf, offset=offset
            )
            view[:] = array

    def close(self):
        """Close the shared memory segment (but don't unlink it)."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        self.lock.close()

    def unlink(self):
        """Unlink (delete) the shared memory segment."""
        if self.shm is not None:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                # Already unlinked
                pass
            self.shm = None
        self.lock.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class ShmIPCManager:
    """Manager for multiple shared memory IPC segments."""

    def __init__(self, lock_file_path: Optional[str] = None):
        """
        Initialize the manager.

        Args:
            lock_file_path: Path to the lock file. If None, uses a default path.
        """
        self.lock_file_path = lock_file_path
        self._segments: dict[str, ShmIPC] = {}

    def create_segment(
        self,
        name: str,
        size: int,
        lock_file_path: Optional[str] = None,
    ) -> ShmIPC:
        """
        Create a new shared memory segment.

        Args:
            name: Name of the shared memory segment.
            size: Size of the segment in bytes.
            lock_file_path: Optional custom lock file path for this segment.

        Returns:
            ShmIPC instance.
        """
        if name in self._segments:
            raise ValueError(f"Segment '{name}' already exists")

        lock_path = lock_file_path or self.lock_file_path
        segment = ShmIPC(
            name=name,
            size=size,
            create=True,
            lock_file_path=lock_path,
        )
        self._segments[name] = segment
        return segment

    def attach_segment(
        self,
        name: str,
        size: int,
        lock_file_path: Optional[str] = None,
    ) -> ShmIPC:
        """
        Attach to an existing shared memory segment.

        Args:
            name: Name of the shared memory segment.
            size: Expected size of the segment in bytes.
            lock_file_path: Optional custom lock file path for this segment.

        Returns:
            ShmIPC instance.
        """
        if name in self._segments:
            return self._segments[name]

        lock_path = lock_file_path or self.lock_file_path
        segment = ShmIPC(
            name=name,
            size=size,
            create=False,
            lock_file_path=lock_path,
        )
        self._segments[name] = segment
        return segment

    def get_segment(self, name: str) -> Optional[ShmIPC]:
        """Get a segment by name."""
        return self._segments.get(name)

    def remove_segment(self, name: str, unlink: bool = False):
        """
        Remove a segment from the manager.

        Args:
            name: Name of the segment.
            unlink: If True, also unlink (delete) the shared memory segment.
        """
        if name in self._segments:
            segment = self._segments[name]
            if unlink:
                segment.unlink()
            else:
                segment.close()
            del self._segments[name]

    def close_all(self, unlink: bool = False):
        """
        Close all segments.

        Args:
            unlink: If True, also unlink (delete) all shared memory segments.
        """
        for name in list(self._segments.keys()):
            self.remove_segment(name, unlink=unlink)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all(unlink=False)


class ShmIpcMemoryChunk:
    """Memory chunk for shared memory IPC pool."""

    def __init__(self, area: Tuple[int, int], sync_buffer: ShmIPC):
        """
        Initialize memory chunk.

        Args:
            area: Tuple of (start, end) offsets in the memory pool.
            sync_buffer: ShmIPC instance for synchronization.
        """
        self.area = area
        self.sync_buffer = sync_buffer

    @property
    def mem_size(self):
        """Get memory size of this chunk."""
        return self.area[1] - self.area[0]

    @property
    def start(self):
        """Get start offset."""
        return self.area[0]

    @property
    def end(self):
        """Get end offset."""
        return self.area[1]

    def try_to_recycle(self, expected_count: int) -> bool:
        """
        Try to recycle this chunk if all processes have finished using it.

        Args:
            expected_count: Expected number of processes using this chunk.

        Returns:
            True if chunk can be recycled, False otherwise.
        """
        try:
            # Read the sync flag from shared memory with lock
            with self.sync_buffer.locked_access():
                sync_value = self.sync_buffer.read_numpy(
                    np.int32, (1,), offset=0, copy=True
                )[0]
                if sync_value >= expected_count:
                    # Reset sync flag
                    self.sync_buffer.write_numpy(
                        np.array([0], dtype=np.int32), offset=0
                    )
                    return True
        except Exception as e:
            logger.info(f"Error: Failed to check sync flag: {e}")
        return False


class ShmIpcMemoryPool:
    """Memory pool for shared memory IPC operations."""

    def __init__(self, memory_size: int, base_name: str = "sglang_shm_pool"):
        """
        Initialize shared memory IPC memory pool.

        Args:
            memory_size: Total size of the memory pool in bytes.
            base_name: Base name for shared memory segments.
        """
        self.memory_size = memory_size
        self.base_name = base_name
        self.shm_manager = ShmIPCManager()
        self.sync_buffers: list[ShmIPC] = []

        # Create main memory pool in shared memory
        self.shm_pool = self.shm_manager.create_segment(
            name=f"{base_name}_main", size=memory_size
        )

        # Initialize available and occupied chunks
        init_chunk = ShmIpcMemoryChunk(
            (0, memory_size), self._pop_sync_buffer()
        )
        self.available_chunks = [init_chunk]
        self.occupied_chunks = []

    def _pop_sync_buffer(self) -> ShmIPC:
        """Get a sync buffer from the pool or create a new one."""
        if len(self.sync_buffers) == 0:
            try:
                sync_buffer = self.shm_manager.create_segment(
                    name=f"{self.base_name}_sync_{len(self.sync_buffers)}",
                    size=4,  # 4 bytes for int32 sync flag
                )
                # Initialize sync flag to 0
                sync_buffer.write_numpy(np.array([0], dtype=np.int32), offset=0)
                return sync_buffer
            except Exception as e:
                logger.info(f"Error: Failed to create sync buffer: {e}")
                raise RuntimeError("Failed to allocate sync buffer")
        else:
            return self.sync_buffers.pop()

    def _push_sync_buffer(self, sync_buffer: ShmIPC):
        """Return a sync buffer to the pool."""
        self.sync_buffers.append(sync_buffer)

    def get_available_chunk(self, src_tensor: torch.Tensor) -> Optional[ShmIpcMemoryChunk]:
        """
        Get an available chunk from the pool for the given tensor.

        Args:
            src_tensor: Source tensor to find a chunk for.

        Returns:
            ShmIpcMemoryChunk if available, None otherwise.
        """
        src_tensor_size = src_tensor.numel() * src_tensor.element_size()
        min_size = self.memory_size + 1
        selected_chunk = None

        for chunk in self.available_chunks:
            if chunk.mem_size >= src_tensor_size:
                if chunk.mem_size < min_size:
                    min_size = chunk.mem_size
                    selected_chunk = chunk

        if selected_chunk:
            occupied_chunk_area = (
                selected_chunk.start,
                selected_chunk.start + src_tensor_size,
            )
            occupied_chunk_sync_flag = selected_chunk.sync_buffer
            new_occupied_chunk = ShmIpcMemoryChunk(
                occupied_chunk_area, occupied_chunk_sync_flag
            )

            self.occupied_chunks.append(new_occupied_chunk)
            self.available_chunks.remove(selected_chunk)

            # Add remaining space as a new available chunk
            available_split_chunk_area = (new_occupied_chunk.end, selected_chunk.end)
            if available_split_chunk_area[0] != available_split_chunk_area[1]:
                split_available_chunk = ShmIpcMemoryChunk(
                    available_split_chunk_area, self._pop_sync_buffer()
                )
                self.available_chunks.append(split_available_chunk)

            return new_occupied_chunk

        return None

    def return_a_slice_tensor_with_flag(
        self, src_tensor: torch.Tensor
    ) -> Tuple[Optional[dict], Optional[torch.Tensor]]:
        """
        Get a slice tensor from the pool with sync flag metadata.

        Args:
            src_tensor: Source tensor.

        Returns:
            Tuple of (sync_flag_meta, slice_tensor) or (None, None) if no chunk available.
        """
        self.recycle_chunks()
        self.merge_chunks()

        available_chunk = self.get_available_chunk(src_tensor)
        if available_chunk is not None:
            # Create a tensor view of the shared memory chunk
            slice_tensor = torch.frombuffer(
                self.shm_pool.shm.buf,
                dtype=torch.uint8,
                count=available_chunk.mem_size,
                offset=available_chunk.start,
            ).view(src_tensor.dtype).view(*src_tensor.shape)

            sync_flag_meta = {
                "handle": available_chunk.sync_buffer.name,
                "shm_pool_name": self.shm_pool.name,
                "offset": available_chunk.start,
                "size": available_chunk.mem_size,
                "shape": list(src_tensor.shape),
                "dtype": str(src_tensor.dtype),
            }

            return sync_flag_meta, slice_tensor

        return None, None

    def recycle_chunks(self):
        """Recycle chunks that are no longer in use."""
        try:
            from sglang.srt.server_args import get_global_server_args

            tp_num = get_global_server_args().tp_size
        except Exception:
            logger.info(
                "Error: get_global_server_args has not been initialized, skip this turn's recycle"
            )
            tp_num = -1

        new_occupied_chunks = []
        for chunk in self.occupied_chunks:
            if chunk.try_to_recycle(tp_num):
                self.available_chunks.append(chunk)
            else:
                new_occupied_chunks.append(chunk)
        self.occupied_chunks = new_occupied_chunks

    def merge_chunks(self):
        """Merge adjacent available chunks."""
        merged_chunks = []
        for chunk in sorted(self.available_chunks, key=lambda x: x.start):
            if len(merged_chunks) == 0:
                merged_chunks.append(chunk)
            else:
                if chunk.start == merged_chunks[-1].end:
                    to_merge_chunk = merged_chunks.pop()
                    to_merge_chunk_sync = to_merge_chunk.sync_buffer
                    merged_chunk_area = (to_merge_chunk.start, chunk.end)
                    merged_chunks.append(
                        ShmIpcMemoryChunk(merged_chunk_area, to_merge_chunk_sync)
                    )
                    self._push_sync_buffer(chunk.sync_buffer)
                else:
                    merged_chunks.append(chunk)

        self.available_chunks = merged_chunks

    def close(self):
        """Close all shared memory segments."""
        self.shm_manager.close_all(unlink=False)


class ShmIpcTensorTransportProxy:
    """
    A torch.tensor's proxy used for inter-process data-sharing via shared memory IPC.

    Similar to CudaIpcTensorTransportProxy but uses /dev/shm instead of CUDA IPC.
    """

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        sync_buffer_meta: dict,
    ):
        """
        Initialize shared memory IPC transport proxy.

        Args:
            data: Tensor data stored in shared memory (CPU tensor).
            info_data: Original tensor with metadata (shape, dtype, etc.).
            sync_buffer_meta: Metadata for synchronization buffer.
        """
        if not isinstance(data, torch.Tensor) or not isinstance(
            info_data, torch.Tensor
        ):
            raise TypeError(
                f"Input 'data' and 'info_data' must be torch.Tensor, but got {type(data)} and {type(info_data)}"
            )

        self.data = data
        self.info_data = info_data
        self.sync_buffer_meta = sync_buffer_meta
        self.sync_buffer: Optional[ShmIPC] = None
        self.reconstruct_tensor = None

    @property
    def get_sync_flag(self) -> np.ndarray:
        """Get the synchronization flag from shared memory."""
        if self.sync_buffer is None:
            shm_name = self.sync_buffer_meta["handle"]
            self.sync_buffer = ShmIPC(name=shm_name, size=4, create=False)

        return self.sync_buffer.read_numpy(np.int32, (1,), offset=0, copy=True)

    def close_shm(self):
        """Close the shared memory buffer."""
        if self.sync_buffer is not None:
            self.sync_buffer.close()
            self.sync_buffer = None

    def reconstruct_on_target_device(self, rebuild_device_idx: int) -> torch.Tensor:
        """
        Reconstruct the tensor on the target device from shared memory.

        Args:
            rebuild_device_idx: Target device index.

        Returns:
            Reconstructed tensor on the target device.
        """
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")

        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        try:
            # Read data from shared memory pool
            shm_pool_name = self.sync_buffer_meta["shm_pool_name"]
            offset = self.sync_buffer_meta["offset"]
            size = self.sync_buffer_meta["size"]
            shape = tuple(self.sync_buffer_meta["shape"])
            dtype_str = self.sync_buffer_meta["dtype"]

            # Attach to shared memory pool
            shm_pool = ShmIPC(name=shm_pool_name, size=size + offset, create=False)

            # Read tensor data from shared memory
            tensor_data = shm_pool.read_numpy(
                dtype=dtype_str, shape=shape, offset=offset, copy=True
            )

            # Convert to torch tensor and move to target device
            reconstructed_tensor = torch.from_numpy(tensor_data).to(
                rebuild_device, non_blocking=True
            )

            # Update sync flag with file lock
            with open("/tmp/shm_wr_lock.lock", "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                sync_flag = self.get_sync_flag
                sync_flag[0] += 1
                self.sync_buffer.write_numpy(sync_flag, offset=0)
                fcntl.flock(f, fcntl.LOCK_UN)

            self.close_shm()
            shm_pool.close()

            self.reconstruct_tensor = reconstructed_tensor
            return self.reconstruct_tensor

        except Exception as e:
            logger.info(f"Error: Failed to reconstruct from shared memory IPC ({e}).")
            raise e
