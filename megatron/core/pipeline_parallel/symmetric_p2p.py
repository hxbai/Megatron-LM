# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from typing import Optional

import torch
import torch.distributed._symmetric_memory as symm_mem


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


class SymmPutBuffer:
    def __init__(self, shape, dtype, pp_group, stream):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.stream = stream
        self.buffer = symm_mem.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        symm_mem.rendezvous(self.buffer, group=pp_group)

    def put_signal(self, tensor, dst, hdl):
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.no_grad():
            with torch.cuda.stream(self.stream):
                self.buffer.copy_(tensor)
                symm_mem.put_signal(self.buffer, hdl, dst)

    def wait(self):
        pass

    def reset(self):
        if self.buffer.grad is not None:
            self.buffer.grad.zero_()


class SymmWaitBuffer:
    def __init__(self, shape, dtype, pp_group, stream):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.buffer = symm_mem.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        self.buffer.requires_grad = True
        self.hdl = symm_mem.rendezvous(self.buffer, group=pp_group)
        self.stream = stream
        # Pre-allocate a PERSISTENT gradient buffer once and bind it as the leaf's .grad. Autograd
        # accumulates in-place into an existing leaf .grad, so the gradient storage address stays
        # stable across iterations. Letting autograd lazily realloc .grad each iter makes a
        # full-iteration CUDA graph captured during warmup bake in a .grad address that is
        # freed/moved before the real loop -> graph replay writes to a stale address -> illegal
        # memory access.
        self.grad_buffer = torch.zeros(shape, dtype=dtype, device=torch.cuda.current_device())
        self.buffer.grad = self.grad_buffer

    def get_hdl(self):
        return self.hdl

    def get_buffer(self):
        self.buffer = self.buffer.detach()
        self.buffer.requires_grad = True
        # Re-bind the persistent grad buffer (detach() drops .grad) and zero in-place so the
        # captured graph always reads/writes the same storage.
        self.buffer.grad = self.grad_buffer
        self.grad_buffer.zero_()
        return self.buffer

    def wait_signal(self, recv_from_rank):
        with torch.cuda.stream(self.stream):
            symm_mem.wait_signal(self.hdl, recv_from_rank)
            # grad_buffer is already zeroed by get_buffer(), which _communicate() calls on this
            # same SymmWaitBuffer immediately before this wait (recv_index/wait_index advance in
            # lockstep), on the compute stream and before the microbatch's backward accumulates
            # into .grad. Re-zeroing here is redundant; dropped to remove a per-recv kernel
            # launch from the host critical path (lowers eager-mode rank skew at collectives).

    def wait(self):
        torch.cuda.current_stream().wait_stream(self.stream)

    def reset(self):
        # grad_buffer zeroing is handled per-recv in get_buffer() (compute stream, before each
        # microbatch's backward accumulates into .grad), so this iteration-boundary zero was
        # redundant. Dropped to remove ~num_recv_buffers kernel launches per iteration from the
        # host path. Kept as a no-op for SymmMemBuffer.reset()'s loop / API stability.
        return


class SymmMemBuffer:
    def __init__(self, shape, dtype, pp_group, num_warmup_microbatches_pp0, sym_mem_pool):
        self.shape = shape
        self.dtype = dtype
        self.pp_group = pp_group
        self.sym_mem_pool = sym_mem_pool
        self.recv_from_rank = None
        self.curr_rank_in_pp_group = pp_group.rank()
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        self.send_buffers = []
        self.recv_buffers = []
        self.hdls = []
        self.num_send_buffers = 1
        self.num_recv_buffers = num_warmup_microbatches_pp0
        self.send_index = 0
        self.recv_index = 0
        self.wait_index = 0

        for i in range(self.num_send_buffers):
            self.send_buffers.append(SymmPutBuffer(shape, dtype, self.pp_group, self.send_stream))
        for i in range(self.num_recv_buffers):
            self.recv_buffers.append(SymmWaitBuffer(shape, dtype, self.pp_group, self.recv_stream))
            self.hdls.append(self.recv_buffers[i].get_hdl())

    def get_recv_buffer(self):
        self.recv_index += 1
        return self.recv_buffers[(self.recv_index - 1) % self.num_recv_buffers].get_buffer()

    def wait_signal(self, src, tensor=None):
        assert self.recv_from_rank is None or self.recv_from_rank == src
        self.recv_from_rank = src
        self.recv_buffers[self.wait_index % self.num_recv_buffers].wait_signal(self.recv_from_rank)
        self.wait_index += 1
        return self.recv_buffers[(self.wait_index - 1) % self.num_recv_buffers]

    def reset(self):
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
        hdl = self.hdls[self.send_index % self.num_recv_buffers]
        self.send_buffers[self.send_index % self.num_send_buffers].put_signal(tensor, dst, hdl)
        self.send_index += 1
        return self.send_buffers[(self.send_index - 1) % self.num_send_buffers]

    def join(self):
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.send_stream)
        current_stream.wait_stream(self.recv_stream)
