# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed._symmetric_memory as symm_mem

from megatron.core.utils import is_tensor_storage_reusable, mark_tensor_storage_reusable


@dataclass(frozen=True)
class RecvSlotPlan:
    """Receive slots indexed by the producer's PP rank and directional send count.

    ``send_slots`` maps forward sends. A missing backward map retains the
    separate backward ring; otherwise both maps address a single receive pool.
    """

    num_buffers: int
    send_slots: tuple[tuple[int, ...], ...]
    # If present, both directions use the same physical receive windows.
    backward_send_slots: tuple[tuple[int, ...], ...] | None = None


def _symm_mem_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    symm_buffers: dict,
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    reqs = {}

    if tensor_send_next is not None:
        reqs["send_next"] = symm_buffers['send_next_recv_prev'].put_signal(
            tensor=tensor_send_next, dst=next_pipeline_rank
        )

    if tensor_recv_prev is not None:
        reqs["recv_prev"] = symm_buffers['send_next_recv_prev'].wait_signal(
            src=prev_pipeline_rank, tensor=tensor_recv_prev
        )

    if tensor_send_prev is not None:
        reqs["send_prev"] = symm_buffers['send_prev_recv_next'].put_signal(
            tensor=tensor_send_prev, dst=prev_pipeline_rank
        )

    if tensor_recv_next is not None:
        reqs["recv_next"] = symm_buffers['send_prev_recv_next'].wait_signal(
            src=next_pipeline_rank, tensor=tensor_recv_next
        )

    return reqs


@dataclass(frozen=True)
class SymmPutWork:
    """Completion of one source copy, independent of later sends on the same buffer."""

    copy_done: torch.cuda.Event
    # References/record_stream protect ordinary sources from allocator reuse.
    # Graph/static buffers can be overwritten even while aliases are alive.
    can_defer_source_wait: bool = False

    def wait(self) -> None:
        """Order subsequent compute after this copy, not the remote transfer."""
        torch.cuda.current_stream().wait_event(self.copy_done)


class SymmPutBuffer:
    def __init__(self, shape, dtype, pp_group, stream):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.stream = stream
        self.copy_done = None
        self.buffer = symm_mem.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        symm_mem.rendezvous(self.buffer, group=pp_group)

    def put_signal(self, tensor, dst, hdl, *, record_source=False):
        current_stream = torch.cuda.current_stream()
        # A schedule can still hold the previous send's work when it enqueues
        # this send. Re-recording a shared event would make that old work wait
        # for this copy, accidentally serializing copy and compute again.
        copy_done = torch.cuda.Event()
        self.stream.wait_stream(current_stream)
        with torch.no_grad():
            with torch.cuda.stream(self.stream):
                self.buffer.copy_(tensor)
                if record_source:
                    # Transient gradients may lose their last reference before
                    # the copy finishes. Forward sources retain the existing
                    # copy-event wait before pseudo-deallocation instead (the
                    # schedule may defer that wait across independent compute).
                    tensor.record_stream(self.stream)
                # The caller may reuse/free its source after this copy, without
                # waiting for the subsequent remote transfer to finish.
                copy_done.record(self.stream)
                symm_mem.put_signal(self.buffer, hdl, dst)
        self.copy_done = copy_done
        return SymmPutWork(copy_done, can_defer_source_wait=not is_tensor_storage_reusable(tensor))

    def wait(self):
        if self.copy_done is not None:
            torch.cuda.current_stream().wait_event(self.copy_done)

    def reset(self):
        if self.buffer.grad is not None:
            self.buffer.grad.zero_()


class SymmWaitBuffer:
    def __init__(self, shape, dtype, pp_group, stream, *, requires_grad=True, shared_buffer=None):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.requires_grad = requires_grad
        if shared_buffer is None:
            self.buffer = symm_mem.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            self.hdl = symm_mem.rendezvous(self.buffer, group=pp_group)
        else:
            if (
                tuple(shared_buffer.shape) != tuple(shape)
                or shared_buffer.dtype != dtype
                or shared_buffer.pp_group is not pp_group
            ):
                raise ValueError(
                    'Shared symmetric P2P window has a different shape, dtype or group'
                )
            # Distinct autograd wrappers/receive streams, but exactly the same
            # allocation and registered window. Never re-register an alias.
            self.buffer = shared_buffer.buffer.detach()
            self.hdl = shared_buffer.hdl
        self.buffer.requires_grad = requires_grad
        mark_tensor_storage_reusable(self.buffer)
        self.stream = stream
        # Forward inputs live until their backward, but their gradients do not.
        # Leave .grad unset so autograd can adopt the freshly computed gradient.
        # The schedule takes and clears it after backward, rather than retaining
        # a gradient for every receive slot. Backward receives never need .grad.

    def get_hdl(self):
        return self.hdl

    def get_buffer(self):
        self.buffer = self.buffer.detach()
        self.buffer.requires_grad = self.requires_grad
        return self.buffer

    def wait_signal(self, recv_from_rank):
        with torch.cuda.stream(self.stream):
            symm_mem.wait_signal(self.hdl, recv_from_rank)

    def wait(self):
        torch.cuda.current_stream().wait_stream(self.stream)

    def reset(self):
        return


class SymmMemBuffer:
    def __init__(
        self,
        shape,
        dtype,
        pp_group,
        num_recv_buffers,
        sym_mem_pool,
        *,
        requires_grad=True,
        shared_recv_buffers=None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.sym_mem_pool = sym_mem_pool
        self.requires_grad = requires_grad
        self.recv_from_rank = None
        self.curr_rank_in_pp_group = pp_group.rank()
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        self.send_buffers = []
        self.recv_buffers = []
        self.hdls = []
        self.num_send_buffers = 1
        self.num_recv_buffers = num_recv_buffers
        self.send_index = 0
        self.recv_index = 0
        self.wait_index = 0
        self.send_slots = None
        self.recv_slots = None
        self.shared_recv_buffers = shared_recv_buffers is not None
        if shared_recv_buffers is not None and len(shared_recv_buffers) != num_recv_buffers:
            raise ValueError('Shared symmetric P2P receive pool has a different capacity')

        for i in range(self.num_send_buffers):
            self.send_buffers.append(SymmPutBuffer(shape, dtype, self.pp_group, self.send_stream))
        for i in range(self.num_recv_buffers):
            self.recv_buffers.append(
                SymmWaitBuffer(
                    shape,
                    dtype,
                    self.pp_group,
                    self.recv_stream,
                    requires_grad=requires_grad,
                    shared_buffer=None if shared_recv_buffers is None else shared_recv_buffers[i],
                )
            )
            self.hdls.append(self.recv_buffers[i].get_hdl())
        self.recv_buffer_ptrs = {buffer.buffer.data_ptr() for buffer in self.recv_buffers}

    def prepare_grad(self, tensor: torch.Tensor) -> bool:
        """Let autograd create/adopt this received input's gradient without zero/add.

        The caller must first wait for the previous gradient's staging copy.
        A CUDA graph may reuse the storage of that previous gradient even though
        the receive tensor no longer holds a reference to it.
        """
        if not self.requires_grad or tensor.data_ptr() not in self.recv_buffer_ptrs:
            return False
        tensor.grad = None
        tensor._symmetric_p2p_input = True
        return True

    def get_recv_buffer(self):
        slot = self._get_slot(self.recv_index, self.recv_slots)
        self.recv_index += 1
        return self.recv_buffers[slot].get_buffer()

    def _get_slot(self, index: int, slots: tuple[int, ...] | None) -> int:
        if slots is None:
            return index % self.num_recv_buffers
        if index >= len(slots):
            raise RuntimeError('Symmetric P2P communication exceeded the planned schedule')
        return slots[index]

    def wait_signal(self, src, tensor=None):
        assert self.recv_from_rank is None or self.recv_from_rank == src
        self.recv_from_rank = src
        slot = self._get_slot(self.wait_index, self.recv_slots)
        buffer = self.recv_buffers[slot]
        if tensor is not None and tensor.data_ptr() != buffer.buffer.data_ptr():
            raise RuntimeError('Symmetric P2P receive tensor does not match its planned slot')
        buffer.wait_signal(self.recv_from_rank)
        self.wait_index += 1
        return buffer

    def reset(
        self,
        *,
        send_slots: tuple[int, ...] | None = None,
        recv_slots: tuple[int, ...] | None = None,
    ):
        if self.send_slots is not None and self.send_index != len(self.send_slots):
            raise RuntimeError('Symmetric P2P reset before the planned sends completed')
        if self.recv_slots is not None and (
            self.recv_index != len(self.recv_slots) or self.wait_index != len(self.recv_slots)
        ):
            raise RuntimeError('Symmetric P2P reset before the planned receives completed')
        for slots in (send_slots, recv_slots):
            if slots is not None and any(
                slot < 0 or slot >= self.num_recv_buffers for slot in slots
            ):
                raise ValueError('Symmetric P2P slot plan exceeds the registered buffer capacity')
        self.send_slots = send_slots
        self.recv_slots = recv_slots
        self.recv_index = 0
        self.send_index = 0
        self.wait_index = 0
        for i in range(self.num_send_buffers):
            self.send_buffers[i].reset()
        for i in range(self.num_recv_buffers):
            self.recv_buffers[i].reset()
        current_stream = torch.cuda.current_stream()
        self.send_stream.wait_stream(current_stream)
        self.recv_stream.wait_stream(current_stream)

    def put_signal(self, tensor, dst):
        hdl = self.hdls[self._get_slot(self.send_index, self.send_slots)]
        work = self.send_buffers[self.send_index % self.num_send_buffers].put_signal(
            tensor, dst, hdl, record_source=not self.requires_grad
        )
        self.send_index += 1
        return work

    def join(self):
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.send_stream)
        current_stream.wait_stream(self.recv_stream)
