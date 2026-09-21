# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import gc
import os
import weakref
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.hybrid_cp_schedule as hybrid_cp_schedule
import megatron.core.pipeline_parallel.p2p_communication as p2p
import megatron.core.pipeline_parallel.schedules as schedule
import megatron.core.pipeline_parallel.symmetric_p2p as symmetric_p2p
import megatron.core.utils as core_utils
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer.cuda_graphs import (
    _CudagraphReplayNode,
    _GraphStatus,
    convert_schedule_table_to_order,
    get_overlap_moe_expert_parallel_comm_order,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import GraphableMegatronModule
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


@pytest.fixture
def pp_group(mocker):
    group = mocker.Mock()
    group.size.return_value = 4
    group.rank.return_value = 1
    backend = group._get_backend.return_value
    backend._comm_ptr.return_value = 0
    backend.registered_comm = 0

    def eager_connect(device):
        # NCCL keeps the first cached comm but publishes each new one to symm_mem.
        backend.registered_comm += 1
        if backend._comm_ptr.return_value == 0:
            backend._comm_ptr.return_value = backend.registered_comm

    backend.eager_connect_single_device.side_effect = eager_connect
    mocker.patch.object(p2p.dist, 'get_global_rank', side_effect=lambda group, rank: rank + 8)
    mocker.patch.object(p2p.dist, 'get_group_rank', side_effect=lambda group, rank: rank - 8)
    return group


@pytest.mark.parametrize('use_symmetric_memory_p2p', [False, True])
@pytest.mark.parametrize('fixed_packed_shape', [False, True])
@pytest.mark.parametrize('already_initialized', [False, True])
def test_eager_initialization_is_scoped_to_symmetric_pp(
    mocker, pp_group, use_symmetric_memory_p2p, fixed_packed_shape, already_initialized
):
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.bfloat16,
        use_symmetric_memory_p2p=use_symmetric_memory_p2p,
        variable_seq_lengths=fixed_packed_shape,
        pipeline_p2p_fixed_shape=fixed_packed_shape,
        sequence_packing_scheduler='dp_balanced' if fixed_packed_shape else None,
        max_seqlen_per_dp_cp_rank=4096 if fixed_packed_shape else None,
        pad_packed_seq_alignment='max' if fixed_packed_shape else None,
    )
    current_device = mocker.patch.object(torch.cuda, 'current_device', return_value=3)
    new_group = mocker.patch.object(p2p.dist, 'new_group')
    set_backend = mocker.patch.object(p2p.symm_mem, 'set_backend')
    get_mem_pool = mocker.patch.object(p2p.symm_mem, 'get_mem_pool')
    backend = pp_group._get_backend.return_value
    backend._comm_ptr.return_value = 1 if already_initialized else 0
    backend.registered_comm = backend._comm_ptr.return_value
    initialization = mocker.Mock()
    initialization.attach_mock(backend._comm_ptr, 'comm_ptr')
    initialization.attach_mock(backend.eager_connect_single_device, 'connect')
    initialization.attach_mock(set_backend, 'set_backend')
    initialization.attach_mock(get_mem_pool, 'get_mem_pool')

    communicator = p2p.P2PCommunicator(pp_group, config)

    new_group.assert_not_called()
    if use_symmetric_memory_p2p:
        device = torch.device('cuda', 3)
        current_device.assert_called_once_with()
        pp_group._get_backend.assert_called_once_with(device)
        expected_calls = [mocker.call.comm_ptr()]
        if not already_initialized:
            expected_calls.append(mocker.call.connect(device))
        assert initialization.mock_calls == expected_calls + [
            mocker.call.set_backend('NCCL'),
            mocker.call.get_mem_pool(device),
        ]
        assert communicator.symm_mem_pool is get_mem_pool.return_value
    else:
        current_device.assert_not_called()
        pp_group._get_backend.assert_not_called()
        assert initialization.mock_calls == []
        assert communicator.symm_mem_pool is None


@pytest.mark.parametrize('graph_helper_first', [False, True])
@pytest.mark.parametrize('already_initialized', [False, True])
def test_symmetric_pp_reuses_communicator_and_windows(
    mocker, pp_group, graph_helper_first, already_initialized
):
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, pipeline_dtype=torch.bfloat16, use_symmetric_memory_p2p=True
    )
    mocker.patch.object(torch.cuda, 'current_device', return_value=3)
    mocker.patch.object(p2p.symm_mem, 'set_backend')
    mocker.patch.object(p2p.symm_mem, 'get_mem_pool')
    mocker.patch.object(p2p.P2PCommunicator, 'symm_buffers', {})
    backend = pp_group._get_backend.return_value
    backend._comm_ptr.return_value = 1 if already_initialized else 0
    backend.registered_comm = backend._comm_ptr.return_value
    create_buffer = mocker.patch.object(
        p2p,
        'SymmMemBuffer',
        side_effect=lambda shape, dtype, group, count, pool, **kwargs: mocker.Mock(
            shape=shape,
            dtype=dtype,
            pp_group=group,
            num_recv_buffers=count,
            shared_recv_buffers=kwargs.get('shared_recv_buffers') is not None,
            registered_comm=backend.registered_comm,
        ),
    )

    # TECudaGraphHelper can construct a communicator before the first schedule.
    # The schedule then constructs another communicator on every iteration.
    if graph_helper_first:
        p2p.P2PCommunicator(pp_group, config)
    windows = None
    for _ in range(3):
        communicator = p2p.P2PCommunicator(pp_group, config)
        communicator.create_symm_put_wait_buffers((16, 1, 8), 2, 4)
        if windows is None:
            windows = dict(communicator.symm_buffers)
        for direction, window in windows.items():
            assert communicator.symm_buffers[direction] is window
            assert window.registered_comm == backend.registered_comm
            assert backend.registered_comm == backend._comm_ptr()
        communicator.symm_join()

    assert create_buffer.call_count == 2
    for window in windows.values():
        assert window.reset.call_count == 3
        assert window.join.call_count == 3
    if already_initialized:
        backend.eager_connect_single_device.assert_not_called()
    else:
        backend.eager_connect_single_device.assert_called_once_with(torch.device('cuda', 3))


def test_symmetric_pp_initialization_is_per_group_and_device(mocker, pp_group):
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, pipeline_dtype=torch.bfloat16, use_symmetric_memory_p2p=True
    )
    current_device = mocker.patch.object(torch.cuda, 'current_device', return_value=0)
    mocker.patch.object(p2p.symm_mem, 'set_backend')
    mocker.patch.object(p2p.symm_mem, 'get_mem_pool')
    other_group = mocker.Mock()
    other_group.size.return_value = pp_group.size()
    other_group.rank.return_value = pp_group.rank()

    for group in (pp_group, other_group):
        backend = group._get_backend.return_value
        comms = {}
        backend._comm_ptr.side_effect = lambda comms=comms: comms.get(
            current_device.return_value, 0
        )

        def eager_connect(device, comms=comms):
            comms[device.index] = comms.get(device.index, 0) + 1

        backend.eager_connect_single_device.side_effect = eager_connect
        for device_index in (0, 0, 1, 1, 0):
            current_device.return_value = device_index
            p2p.P2PCommunicator(group, config)

        assert backend.eager_connect_single_device.call_args_list == [
            mocker.call(torch.device('cuda', 0)),
            mocker.call(torch.device('cuda', 1)),
        ]


@pytest.fixture
def mock_symmetric_memory(mocker):
    """Use real CPU autograd while recording the CUDA stream protocol."""
    trace = []
    regular_allocations = []
    original_empty = torch.empty

    class Stream:
        def wait_stream(self, stream):
            trace.append(('wait_stream', self, stream))

        def wait_event(self, event):
            trace.append(('wait_copy', event))

    class Event:
        def record(self, stream):
            trace.append(('record_copy', self))

    current = [Stream()]

    @contextmanager
    def stream_context(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    def empty(*args, **kwargs):
        kwargs['device'] = 'cpu'
        tensor = original_empty(*args, **kwargs)
        regular_allocations.append(tensor)
        return tensor

    mocker.patch.object(torch.cuda, 'current_device', return_value=0)
    mocker.patch.object(torch.cuda, 'current_stream', side_effect=lambda: current[0])
    mocker.patch.object(torch.cuda, 'Stream', Stream)
    mocker.patch.object(torch.cuda, 'Event', Event)
    mocker.patch.object(torch.cuda, 'stream', stream_context)
    mocker.patch.object(
        torch.Tensor,
        'record_stream',
        lambda tensor, stream: trace.append(('record_source', tensor.data_ptr(), stream)),
    )
    mocker.patch.object(torch, 'empty', side_effect=empty)
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'empty',
        side_effect=lambda shape, dtype, device: original_empty(shape, dtype=dtype, device='cpu'),
        create=True,
    )
    mocker.patch.object(
        symmetric_p2p.symm_mem, 'rendezvous', side_effect=lambda *a, **k: object(), create=True
    )
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'put_signal',
        side_effect=lambda *args: trace.append(('put', args)),
        create=True,
    )
    mocker.patch.object(symmetric_p2p.symm_mem, 'wait_signal', create=True)
    mocker.patch.object(p2p.P2PCommunicator, 'symm_buffers', {})
    return SimpleNamespace(trace=trace, regular_allocations=regular_allocations)


def _make_symmetric_test_communicator(mocker, *, pp=4, vp=4, group_size=4, overlap_ep=False):
    communicator = object.__new__(p2p.P2PCommunicator)
    communicator.pp_group = mocker.Mock()
    communicator.pp_group.size.return_value = pp
    communicator.pp_group.rank.return_value = 1
    communicator.config = ModelParallelConfig(
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        microbatch_group_size_per_vp_stage=group_size,
        overlap_moe_expert_parallel_comm=overlap_ep,
        pipeline_dtype=torch.float32,
        use_symmetric_memory_p2p=True,
    )
    communicator.virtual_pipeline_model_parallel_size = vp
    communicator.symm_mem_pool = None
    return communicator


@pytest.mark.parametrize(
    'pp,vp,group_size,overlap_ep,forward_slots,backward_slots',
    [
        (4, 4, 4, False, 31, 7),
        (8, 4, 8, False, 63, 15),
        (4, None, 4, False, 4, 4),
        (4, 4, 8, False, 55, 55),
        (4, 4, 4, True, 32, 32),
    ],
)
@pytest.mark.parametrize('slot_plan', ['none', 'forward', 'shared'])
def test_symmetric_pp_buffer_budget(
    mocker,
    mock_symmetric_memory,
    pp,
    vp,
    group_size,
    overlap_ep,
    forward_slots,
    backward_slots,
    slot_plan,
):
    communicator = _make_symmetric_test_communicator(
        mocker, pp=pp, vp=vp, group_size=group_size, overlap_ep=overlap_ep
    )
    plan = None
    if slot_plan != 'none':
        planner = (
            schedule._get_symmetric_recv_slot_plan
            if slot_plan == 'shared'
            else schedule._get_symmetric_forward_slot_plan
        )
        plan = planner(pp, vp or 1, 64, group_size, overlap_moe_expert_parallel_comm=overlap_ep)
    communicator.create_symm_put_wait_buffers((2, 1, 4), forward_slots - 1, 64, recv_slot_plan=plan)
    if plan is not None:
        forward_slots = plan.num_buffers
        if plan.backward_send_slots is not None:
            backward_slots = forward_slots
    forward = communicator.symm_buffers['send_next_recv_prev']
    backward = communicator.symm_buffers['send_prev_recv_next']
    assert forward.num_recv_buffers == forward_slots
    assert backward.num_recv_buffers == backward_slots
    assert all(
        buffer.buffer.grad is None for buffer in forward.recv_buffers + backward.recv_buffers
    )
    assert all(not buffer.buffer.requires_grad for buffer in backward.recv_buffers)
    assert mock_symmetric_memory.regular_allocations == []
    shared_pool = plan is not None and plan.backward_send_slots is not None
    physical_slots = len(
        {buffer.buffer.data_ptr() for buffer in forward.recv_buffers + backward.recv_buffers}
    )
    assert physical_slots == (forward_slots if shared_pool else forward_slots + backward_slots)
    # Only the two staging sends and physical receive windows are registered.
    assert symmetric_p2p.symm_mem.empty.call_count == physical_slots + 2
    assert symmetric_p2p.symm_mem.rendezvous.call_count == physical_slots + 2
    if shared_pool:
        assert forward.recv_stream is not backward.recv_stream
        for fwd, bwd in zip(forward.recv_buffers, backward.recv_buffers):
            assert fwd.hdl is bwd.hdl
            assert fwd.buffer is not bwd.buffer
            assert fwd.buffer.requires_grad and not bwd.buffer.requires_grad

    communicator.prepare_input_tensor_grad(forward.get_recv_buffer())
    assert mock_symmetric_memory.regular_allocations == []
    assert all(buffer.buffer.grad is None for buffer in forward.recv_buffers)
    if (pp, vp, group_size, overlap_ep) == (4, 4, 4, False):
        # Persistent payload only: 22 shared receives and two staging sends.
        # Autograd's transient input gradient is not retained in the receive pool.
        expected_gib = {'none': 8.75, 'forward': 6.125, 'shared': 5.25}
        assert (physical_slots + 2) * 224 / 1024 == expected_gib[slot_plan]


@pytest.mark.parametrize('deallocate_outputs', [False, True])
def test_symmetric_pp_transient_gradients_match_autograd(
    mocker, mock_symmetric_memory, deallocate_outputs
):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.config.deallocate_pipeline_outputs = deallocate_outputs
    shape = (2, 1, 4)
    for iteration in range(3):
        communicator.create_symm_put_wait_buffers(shape, 30, 64)
        forward = communicator.symm_buffers['send_next_recv_prev']
        backward = communicator.symm_buffers['send_prev_recv_next']
        # Wrap both rings, with inputs retained and backward visiting them out of order.
        for group in range(9):
            inputs, outputs, values = [], [], []
            for i in range(4):
                tensor = forward.get_recv_buffer()
                assert tensor.grad is None
                value = 1 + iteration + group + i
                with torch.no_grad():
                    tensor.fill_(value)
                inputs.append(tensor)
                outputs.append(tensor.square())
                values.append(value)
            for i in (2, 0, 3, 1):
                upstream = backward.get_recv_buffer()
                upstream.fill_(3)
                assert not upstream.requires_grad and upstream.grad is None
                communicator.prepare_input_tensor_grad(inputs[i])
                assert inputs[i].grad is None
                schedule.deallocate_output_tensor(outputs[i], deallocate_outputs)
                grad = schedule.backward_step(inputs[i], outputs[i], upstream, communicator.config)
                assert inputs[i].grad is None
                torch.testing.assert_close(grad, torch.full_like(inputs[i], 6 * values[i]))
                backward.put_signal(grad, dst=0)
                torch.testing.assert_close(backward.send_buffers[0].buffer, grad)
                assert (
                    'record_source',
                    grad.data_ptr(),
                    backward.send_stream,
                ) in mock_symmetric_memory.trace
                grad_ref = weakref.ref(grad)
                del grad
                assert grad_ref() is None
        communicator.symm_join()
        assert all(buffer.buffer.grad is None for buffer in forward.recv_buffers)
    allocations = mock_symmetric_memory.regular_allocations
    # Pseudo-deallocation replaces each output's storage with a scalar. No
    # allocation here should be an input-sized persistent gradient buffer.
    assert len(allocations) == (3 * 9 * 4 if deallocate_outputs else 0)
    assert all(tensor.numel() == 1 for tensor in allocations)


def test_symmetric_pp_waits_for_copy_before_next_backward(mocker, mock_symmetric_memory):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64)
    forward = communicator.symm_buffers['send_next_recv_prev']
    backward = communicator.symm_buffers['send_prev_recv_next']
    tensor = forward.get_recv_buffer()
    communicator.prepare_input_tensor_grad(tensor)
    grad = torch.full_like(tensor, 5)
    tensor.grad = grad
    backward.put_signal(grad, dst=0)
    trace = mock_symmetric_memory.trace
    assert [entry[0] for entry in trace][-2:] == ['record_copy', 'put']
    original_prepare = forward.prepare_grad

    def prepare_after_wait(tensor):
        assert trace[-1] == ('wait_copy', backward.send_buffers[0].copy_done)
        return original_prepare(tensor)

    mocker.patch.object(forward, 'prepare_grad', side_effect=prepare_after_wait)
    communicator.prepare_input_tensor_grad(tensor)
    assert tensor.grad is None
    torch.testing.assert_close(grad, torch.full_like(tensor, 5))
    torch.testing.assert_close(backward.send_buffers[0].buffer, torch.full_like(tensor, 5))


@pytest.mark.parametrize('requires_grad', [False, True])
def test_symmetric_pp_send_work_is_per_copy(mocker, mock_symmetric_memory, requires_grad):
    group = mocker.Mock()
    group.rank.return_value = 1
    buffers = symmetric_p2p.SymmMemBuffer(
        (2, 1, 4), torch.float32, group, 2, None, requires_grad=requires_grad
    )
    trace = mock_symmetric_memory.trace
    first = buffers.put_signal(torch.ones(2, 1, 4), dst=0)
    second = buffers.put_signal(torch.full((2, 1, 4), 2.0), dst=0)
    assert first is not second
    assert first.copy_done is not second.copy_done
    # This is the schedule's actual ordering: enqueue new, then wait old.
    # Waiting an old work must not depend on the newly enqueued copy.
    first.wait()
    assert trace[-1] == ('wait_copy', first.copy_done)
    second.wait()
    assert trace[-1] == ('wait_copy', second.copy_done)
    buffers.send_buffers[0].wait()
    assert trace[-1] == ('wait_copy', second.copy_done)
    assert [entry[0] for entry in trace if entry[0] in ('record_copy', 'put')] == [
        'record_copy',
        'put',
        'record_copy',
        'put',
    ]
    assert len(buffers.send_buffers) == 1
    assert mock_symmetric_memory.regular_allocations == []
    buffers.reset()
    third = buffers.put_signal(torch.ones(2, 1, 4), dst=0)
    first.wait()
    assert trace[-1] == ('wait_copy', first.copy_done)
    assert third.copy_done not in (first.copy_done, second.copy_done)


@pytest.mark.parametrize('alias_kind', ['detach', 'view', 'viewless'])
def test_symmetric_pp_reusable_storage_survives_aliases(alias_kind):
    source = torch.ones(4)
    core_utils.mark_tensor_storage_reusable({'outputs': [source, None]})
    storage_ref = weakref.ref(source.untyped_storage())
    if alias_kind == 'detach':
        alias = source.detach()
    elif alias_kind == 'view':
        alias = source[1:]
    else:
        alias = core_utils.make_viewless_tensor(source[1:], requires_grad=False, keep_graph=False)
    del source
    gc.collect()
    assert core_utils.is_tensor_storage_reusable(alias)
    assert not core_utils.is_tensor_storage_reusable(alias.clone())
    assert not core_utils.is_tensor_storage_reusable(alias + 1)
    del alias
    gc.collect()
    assert storage_ref() is None, 'Lifetime metadata must not retain the allocation'


@pytest.mark.parametrize('reusable', [False, True])
def test_symmetric_pp_send_checks_actual_storage(mocker, mock_symmetric_memory, reusable):
    sender = symmetric_p2p.SymmPutBuffer((4,), torch.float32, mocker.Mock(), torch.cuda.Stream())
    tensor = torch.ones(4)
    if reusable:
        core_utils.mark_tensor_storage_reusable(tensor)
    work = sender.put_signal(tensor.detach(), dst=0, hdl=object())
    assert work.can_defer_source_wait is not reusable
    cloned_work = sender.put_signal(tensor.clone(), dst=0, hdl=object())
    assert cloned_work.can_defer_source_wait
    receiver = symmetric_p2p.SymmWaitBuffer((4,), torch.float32, mocker.Mock(), torch.cuda.Stream())
    received_work = sender.put_signal(receiver.get_buffer(), dst=0, hdl=object())
    assert not received_work.can_defer_source_wait


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('eager_boundary', [False, True])
def test_symmetric_pp_te_replay_marks_outputs_and_gradients(monkeypatch, enabled, eager_boundary):
    """Use the real TE replay wrapper with a CPU stand-in for TE's autograd node."""
    static_output = torch.empty(4)
    static_grad = torch.empty(4)

    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            static_output.copy_(tensor)
            return static_output.detach(), static_output.detach()

        @staticmethod
        def backward(ctx, first, second):
            static_grad.copy_(first + second)
            return static_grad.detach()

    def graphed(tensor, **kwargs):
        first, second = Graphed.apply(tensor)
        return {'outputs': [first, second], 'unused': None}

    module = SimpleNamespace(
        config=SimpleNamespace(use_symmetric_memory_p2p=enabled),
        cuda_graphs=[graphed],
        cuda_graph_manual_hooks=[],
        _get_te_cuda_graph_replay_args=lambda *args, **kwargs: (args, kwargs),
    )
    # A Mock/spy retains its arguments, which itself prevents autograd from
    # adopting returned gradient storage. Count calls without retaining tensors.
    mark_count = 0
    original_mark = core_utils.mark_tensor_storage_reusable

    def mark(tensors):
        nonlocal mark_count
        mark_count += 1
        original_mark(tensors)

    monkeypatch.setattr(core_utils, 'mark_tensor_storage_reusable', mark)
    tensor = torch.ones(4, requires_grad=True)
    graph_input = tensor * 2 if eager_boundary else tensor
    outputs = GraphableMegatronModule._te_cuda_graph_replay(module, graph_input)['outputs']
    assert outputs[0].grad_fn is outputs[1].grad_fn
    assert core_utils.is_tensor_storage_reusable(outputs[0]) is enabled
    assert core_utils.is_tensor_storage_reusable(outputs[1].detach()) is enabled
    assert module._te_cuda_graph_route_replay_state is None
    boundary = outputs[0] + 1 if eager_boundary else outputs[0]
    assert core_utils.is_tensor_storage_reusable(boundary) is (enabled and not eager_boundary)
    (boundary.sum() + outputs[1].sum()).backward()
    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 4 if eager_boundary else 2))
    assert (tensor.grad.data_ptr() == static_grad.data_ptr()) is not eager_boundary
    assert core_utils.is_tensor_storage_reusable(tensor.grad) is (enabled and not eager_boundary)
    # One forward mark and one backward hook, despite two outputs sharing a node.
    assert mark_count == (2 if enabled else 0)


@pytest.mark.parametrize('is_last_layer', [False, True])
@pytest.mark.parametrize('enabled', [None, False, True])
def test_symmetric_pp_local_replay_marks_only_static_outputs(
    mocker, mock_symmetric_memory, is_last_layer, enabled
):
    base_module = SimpleNamespace()
    if enabled is not None:
        base_module.config = SimpleNamespace(use_symmetric_memory_p2p=enabled)
    runner = SimpleNamespace(
        base_module=base_module,
        status=_GraphStatus.FWD_READY,
        fwd_graph=mocker.Mock(),
        bwd_graph=mocker.Mock(),
        fwd_graph_input_surface=(torch.ones(4),),
        fwd_graph_output_surface=(torch.full((4,), 2.0),),
        static_grad_outputs=(torch.zeros(4),),
        static_grad_inputs=(torch.full((4,), 3.0),),
        bwd_graph_replay_complete_event=torch.cuda.Event(),
        params_to_backprop=[],
        fp8_enabled=False,
        fp4_enabled=False,
        is_last_layer=is_last_layer,
    )
    ctx = SimpleNamespace()
    outputs = _CudagraphReplayNode.forward(ctx, runner, True, torch.ones(4))
    runner.fwd_graph.replay.assert_called_once()
    assert core_utils.is_tensor_storage_reusable(outputs[0]) is (
        bool(enabled) and not is_last_layer
    )
    assert (
        outputs[0].data_ptr() == runner.fwd_graph_output_surface[0].data_ptr()
    ) is not is_last_layer
    runner.status = _GraphStatus.BWD_READY
    gradients = _CudagraphReplayNode.backward(ctx, torch.ones(4))
    runner.bwd_graph.replay.assert_called_once()
    assert core_utils.is_tensor_storage_reusable(gradients[2]) is bool(enabled)
    torch.testing.assert_close(gradients[2], torch.full((4,), 3.0))


@pytest.mark.parametrize(
    'overrides,forward_only,expected',
    [
        ({}, False, True),
        ({'cuda_graph_impl': 'none'}, False, True),
        ({'use_symmetric_memory_p2p': False}, False, False),
        ({'overlap_p2p_comm': False}, False, False),
        ({'overlap_moe_expert_parallel_comm': True}, False, False),
        ({}, True, False),
        ({'cuda_graph_impl': 'local'}, False, True),
        ({'cuda_graph_impl': 'full_iteration'}, False, True),
        ({'cuda_graph_modules': []}, False, True),
        ({'cuda_graph_modules': [CudaGraphModule.attn, CudaGraphModule.moe]}, False, True),
        ({'mhc_recompute_attn_cuda_graph_split': False}, False, True),
        ({'enable_hyper_connections': False}, False, True),
        ({'recompute_granularity': None}, False, True),
        ({'recompute_modules': ['mla_up_proj']}, False, True),
        ({'recompute_modules': ['mhc', 'mlp']}, False, True),
    ],
)
def test_symmetric_pp_copy_overlap_guards(overrides, forward_only, expected):
    config = SimpleNamespace(
        use_symmetric_memory_p2p=True,
        overlap_p2p_comm=True,
        overlap_moe_expert_parallel_comm=False,
        cuda_graph_impl='transformer_engine',
        cuda_graph_modules=[CudaGraphModule.attn],
        enable_hyper_connections=True,
        mhc_recompute_attn_cuda_graph_split=True,
        recompute_granularity='selective',
        recompute_modules=['mhc', 'mla_up_proj'],
    )
    vars(config).update(overrides)
    assert schedule._can_overlap_symmetric_p2p_copy(config, forward_only=forward_only) is expected


@pytest.mark.parametrize('pp_rank', range(4))
@pytest.mark.parametrize('num_microbatches', [4, 8])
@pytest.mark.parametrize('warmup_overlap', [False, True])
@pytest.mark.parametrize('deallocate_outputs', [False, True])
@pytest.mark.parametrize('graph_mode', ['none', 'split', 'whole_layer'])
@pytest.mark.parametrize('reusable_forward', [False, True])
@pytest.mark.parametrize('reusable_backward', [False, True])
def test_symmetric_pp_copy_overlap_in_interleaved_schedule(
    mocker,
    mock_symmetric_memory,
    pp_rank,
    num_microbatches,
    warmup_overlap,
    deallocate_outputs,
    graph_mode,
    reusable_forward,
    reusable_backward,
):
    """Run the real schedule/autograd, mocking transport and CUDA stream submission only."""
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
        microbatch_group_size_per_vp_stage=4,
        pipeline_dtype=torch.float32,
        use_symmetric_memory_p2p=True,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=warmup_overlap,
        deallocate_pipeline_outputs=deallocate_outputs,
    )
    config.hidden_size = 4
    config.cuda_graph_impl = 'none' if graph_mode == 'none' else 'transformer_engine'
    config.cuda_graph_modules = [CudaGraphModule.attn] if graph_mode == 'split' else []
    config.enable_hyper_connections = graph_mode == 'split'
    config.num_residual_streams = 1
    config.mhc_recompute_attn_cuda_graph_split = graph_mode == 'split'
    config.recompute_granularity = 'selective'
    config.recompute_modules = ['mhc']
    shape = (2, 1, 4)
    trace = mock_symmetric_memory.trace
    phase = [None]
    comm = mocker.Mock(config=config, virtual_pipeline_model_parallel_size=2)
    comm.pp_group.size.return_value = 4
    comm.pp_group.rank.return_value = pp_rank
    senders = {
        direction: symmetric_p2p.SymmPutBuffer(
            shape, torch.float32, comm.pp_group, torch.cuda.Stream()
        )
        for direction in ('f', 'b')
    }
    sends = []
    real_zeros = torch.zeros
    real_backward = schedule.backward_step
    real_deallocate = schedule.deallocate_output_tensor
    mocker.patch.object(torch.distributed, 'is_initialized', return_value=True)

    def has_waited(work):
        return ('wait_copy', work.copy_done) in trace

    def send_recv(tensor, recv, overlap, direction):
        requests = {}
        if tensor is not None:
            if direction == 'f':
                # No growing queue of pending forward sources, including when
                # pseudo-deallocation is disabled or the physical rank wraps VPP.
                assert all(has_waited(work) for d, _, work, _ in sends if d == 'f')
            work = senders[direction].put_signal(
                tensor, dst=0, hdl=object(), record_source=direction == 'b'
            )
            sends.append((direction, phase[0], work, weakref.ref(tensor)))
            trace.append(('send_' + direction, work))
            requests['send_next' if direction == 'f' else 'send_prev'] = work
        incoming = torch.ones(shape, requires_grad=direction == 'f') if recv else None
        if recv:
            requests['recv_prev' if direction == 'f' else 'recv_next'] = mocker.Mock()
        if overlap:
            return incoming, requests
        for request in requests.values():
            request.wait()
        return incoming

    comm.send_forward_recv_forward.side_effect = (
        lambda output_tensor, recv_prev, tensor_shape, overlap_p2p_comm=False: send_recv(
            output_tensor, recv_prev, overlap_p2p_comm, 'f'
        )
    )
    comm.send_backward_recv_backward.side_effect = (
        lambda input_tensor_grad, recv_next, tensor_shape, overlap_p2p_comm=False: send_recv(
            input_tensor_grad, recv_next, overlap_p2p_comm, 'b'
        )
    )
    comm.recv_forward.side_effect = lambda *a, **k: torch.ones(shape, requires_grad=True)
    comm.recv_backward.side_effect = lambda *a, **k: torch.ones(shape)
    comm.prepare_input_tensor_grad.side_effect = lambda tensor: senders['b'].wait()

    def forward(step_func, iterator, model, count, input_tensor, store, config, **kwargs):
        trace.append(('compute_f',))
        if input_tensor is None:
            input_tensor = torch.ones(shape, requires_grad=True)
        output = input_tensor.square()
        if reusable_forward:
            core_utils.mark_tensor_storage_reusable(output)
        return (output.sum() if kwargs['is_last_stage'] else output), 0

    def backward(*args):
        trace.append(('compute_b',))
        gradients = real_backward(*args)
        if reusable_backward:
            core_utils.mark_tensor_storage_reusable(gradients)
        return gradients

    def deallocate(tensor, enabled):
        if tensor is not None and enabled and tensor.numel() > 1:
            for direction, _, work, source in sends:
                if direction == 'f' and source() is tensor:
                    assert has_waited(work), 'Source freed before its staging copy'
        return real_deallocate(tensor, enabled)

    mocker.patch.object(schedule, 'get_model_config', return_value=config)
    mocker.patch.object(schedule, 'get_model_type', return_value=None)
    mocker.patch.object(schedule, 'forward_step', side_effect=forward)
    mocker.patch.object(schedule, 'backward_step', side_effect=backward)
    mocker.patch.object(schedule, 'deallocate_output_tensor', side_effect=deallocate)
    mocker.patch.object(
        schedule, 'nvtx_range_push', side_effect=lambda suffix: phase.__setitem__(0, suffix)
    )
    mocker.patch.object(schedule, 'nvtx_range_pop')
    mocker.patch.object(
        torch, 'zeros', side_effect=lambda *a, **k: real_zeros(*a, **{**k, 'device': 'cpu'})
    )
    group = mocker.Mock()
    group.size.return_value = 1
    schedule.forward_backward_pipelining_with_interleaving(
        forward_step_func=None,
        data_iterator=[iter(()) for _ in range(2)],
        model=[torch.nn.Linear(4, 4) for _ in range(2)],
        num_microbatches=num_microbatches,
        seq_length=2,
        micro_batch_size=1,
        p2p_communicator=comm,
        pg_collection=SimpleNamespace(tp=group, cp=group),
    )
    for direction, send_phase, work, source in sends:
        assert has_waited(work), 'Final copy was not drained'
        send_index = trace.index(('send_' + direction, work))
        wait_index = trace.index(('wait_copy', work.copy_done))
        assert wait_index > send_index
        if direction == 'f' and send_phase == 'steady':
            next_backward = next(
                i for i in range(send_index + 1, len(trace)) if trace[i][0] == 'compute_b'
            )
            if reusable_forward:
                assert wait_index < next_backward
            else:
                assert next_backward < wait_index, 'Forward copy serialized backward'
        if direction == 'b' and send_phase == 'steady':
            next_compute = next(
                (
                    i
                    for i in range(send_index + 1, len(trace))
                    if trace[i][0].startswith('compute_')
                ),
                None,
            )
            if next_compute is not None and trace[next_compute][0] == 'compute_f':
                if reusable_backward:
                    assert wait_index < next_compute
                else:
                    assert next_compute < wait_index, 'Backward copy serialized forward'


def test_symmetric_pp_multiple_inputs_do_not_alias_gradients(mocker, mock_symmetric_memory):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64)
    forward = communicator.symm_buffers['send_next_recv_prev']
    first, second = forward.get_recv_buffer(), forward.get_recv_buffer()
    with torch.no_grad():
        first.fill_(2)
        second.fill_(3)
    other = torch.ones_like(first, requires_grad=True)
    communicator.prepare_input_tensor_grad({'a': [first], 'b': second, 'c': [None, other]})
    assert first.grad is None and second.grad is None and other.grad is None
    outputs = {'a': first.square().sum(), 'b': second.square().sum(), 'c': other.sum()}
    grads = schedule.backward_step_multimodule(
        {'a': [first], 'b': second, 'c': other}, outputs, {}, communicator.config, 'a'
    )
    assert grads['a'].data_ptr() != grads['b'].data_ptr()
    torch.testing.assert_close(grads['a'], torch.full_like(first, 4))
    torch.testing.assert_close(grads['b'], torch.full_like(second, 6))
    assert first.grad is None and second.grad is None
    assert other.grad is grads['c']
    assert mock_symmetric_memory.regular_allocations == []


def test_symmetric_pp_gradient_transfer_avoids_zero_and_add(mocker, mock_symmetric_memory):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64)
    forward = communicator.symm_buffers['send_next_recv_prev']
    tensor = forward.get_recv_buffer()
    with torch.no_grad():
        tensor.fill_(3)
    output = tensor.square().sum()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        communicator.prepare_input_tensor_grad(tensor)
        grad = schedule.backward_step(tensor, output, None, communicator.config)
    operators = {event.key for event in profile.key_averages()}
    assert 'aten::zero_' not in operators
    assert 'aten::add_' not in operators
    assert tensor.grad is None
    torch.testing.assert_close(grad, torch.full_like(tensor, 6))


@pytest.mark.parametrize('participation', ['used', 'unused', 'no_backward'])
@pytest.mark.parametrize('multimodule', [False, True])
def test_symmetric_pp_gradient_transfer_preserves_unused_inputs(
    mocker, mock_symmetric_memory, participation, multimodule
):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64)
    tensor = communicator.symm_buffers['send_next_recv_prev'].get_recv_buffer()
    with torch.no_grad():
        tensor.fill_(2)
    communicator.prepare_input_tensor_grad(tensor)
    if participation == 'used':
        output = tensor.square().sum()
    else:
        output = torch.ones((), requires_grad=participation == 'unused')
    if multimodule:
        grad = schedule.backward_step_multimodule(
            {'input': tensor}, {'input': output}, {}, communicator.config, 'input'
        )['input']
    else:
        grad = schedule.backward_step(tensor, output, None, communicator.config)
    assert tensor.grad is None
    assert not grad.requires_grad
    torch.testing.assert_close(grad, torch.full_like(tensor, 4 if participation == 'used' else 0))


def test_non_symmetric_pp_keeps_input_gradient(mocker):
    config = ModelParallelConfig(pipeline_dtype=torch.float32)
    tensor = torch.ones((2, 1, 4), requires_grad=True)
    output = tensor.square().sum()
    grad = schedule.backward_step(tensor, output, None, config)
    assert grad is tensor.grad
    torch.testing.assert_close(grad, torch.full_like(tensor, 2))


@pytest.mark.parametrize('multimodule', [False, True])
def test_symmetric_pp_gradient_transfer_preserves_aliased_inputs(
    mocker, mock_symmetric_memory, multimodule
):
    communicator = _make_symmetric_test_communicator(mocker)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64)
    tensor = communicator.symm_buffers['send_next_recv_prev'].get_recv_buffer()
    with torch.no_grad():
        tensor.fill_(2)
    communicator.prepare_input_tensor_grad([tensor, tensor])
    if multimodule:
        grads = schedule.backward_step_multimodule(
            {'a': tensor, 'b': [tensor]},
            {'a': tensor.square().sum(), 'b': (tensor * 3).sum()},
            {},
            communicator.config,
            'a',
        )
        first, second = grads['a'], grads['b']
    else:
        first, second = schedule.backward_step(
            [tensor, tensor], tensor.square().sum(), None, communicator.config
        )
    assert tensor.grad is None
    assert first is second
    torch.testing.assert_close(first, torch.full_like(tensor, 7 if multimodule else 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA graph capture')
@pytest.mark.parametrize('slot_plan', ['none', 'forward', 'shared'])
def test_symmetric_pp_transient_gradients_cuda_graph_replay(mocker, slot_plan):
    """Exercise real CUDA streams/graphs, replacing only the remote transport."""
    shape = (2, 1, 4)
    mocker.patch.object(p2p.P2PCommunicator, 'symm_buffers', {})
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'empty',
        side_effect=lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
        create=True,
    )
    mocker.patch.object(
        symmetric_p2p.symm_mem, 'rendezvous', side_effect=lambda *a, **k: object(), create=True
    )
    values = (2, 3, 4)
    received = torch.empty((len(values), *shape), device='cuda')
    num_puts = 0

    def put_signal(tensor, handle, peer):
        nonlocal num_puts
        received[num_puts % len(values)].copy_(tensor)
        num_puts += 1

    mocker.patch.object(symmetric_p2p.symm_mem, 'put_signal', side_effect=put_signal, create=True)
    mocker.patch.object(symmetric_p2p.symm_mem, 'wait_signal', create=True)
    communicator = _make_symmetric_test_communicator(mocker)
    plan = (
        symmetric_p2p.RecvSlotPlan(
            3,
            ((2, 0, 2), (), (), ()),
            ((), (0, 2, 0), (0, 2, 0), ()) if slot_plan == 'shared' else None,
        )
        if slot_plan != 'none'
        else None
    )

    def step():
        communicator.create_symm_put_wait_buffers(shape, 30, 64, recv_slot_plan=plan)
        forward = communicator.symm_buffers['send_next_recv_prev']
        backward = communicator.symm_buffers['send_prev_recv_next']
        for value in values:
            tensor = forward.get_recv_buffer()
            forward.wait_signal(src=0, tensor=tensor).wait()
            with torch.no_grad():
                tensor.fill_(value)
            output = tensor.square()
            communicator.prepare_input_tensor_grad(tensor)
            if slot_plan == 'shared':
                grad = backward.get_recv_buffer()
                backward.wait_signal(src=2, tensor=grad).wait()
                grad.fill_(1)
                input_grad = schedule.backward_step(tensor, output, grad, communicator.config)
            else:
                input_grad = schedule.backward_step(tensor, output.sum(), None, communicator.config)
            assert tensor.grad is None
            backward.put_signal(input_grad, dst=0)
        communicator.symm_join()

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    forward = communicator.symm_buffers['send_next_recv_prev']
    assert all(buffer.buffer.grad is None for buffer in forward.recv_buffers)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for index, value in enumerate(values):
            torch.testing.assert_close(received[index], torch.full_like(received[index], 2 * value))
        assert all(buffer.buffer.grad is None for buffer in forward.recv_buffers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires real asynchronous CUDA copies')
def test_symmetric_pp_transient_gradient_copy_keeps_storage_alive(mocker):
    """Drop a gradient while its copy is delayed, then reuse the compute allocator."""
    shape = (256, 256)
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'empty',
        side_effect=lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
        create=True,
    )
    mocker.patch.object(
        symmetric_p2p.symm_mem, 'rendezvous', side_effect=lambda *a, **k: object(), create=True
    )
    received = torch.empty(shape, device='cuda')
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'put_signal',
        side_effect=lambda tensor, handle, peer: received.copy_(tensor),
        create=True,
    )
    group = mocker.Mock()
    group.rank.return_value = 1
    buffers = symmetric_p2p.SymmMemBuffer(shape, torch.float32, group, 1, None, requires_grad=False)
    with torch.cuda.stream(buffers.send_stream):
        torch.cuda._sleep(20_000_000)
    source = torch.full(shape, 7, device='cuda')
    source_ref = weakref.ref(source)
    buffers.put_signal(source, dst=0)
    del source
    assert source_ref() is None
    # These compute-stream allocations must not overwrite the in-flight source.
    for _ in range(16):
        scratch = torch.empty(shape, device='cuda')
        scratch.fill_(-1)
        del scratch
    buffers.join()
    torch.cuda.synchronize()
    torch.testing.assert_close(received, torch.full_like(received, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA streams')
def test_symmetric_pp_deferred_forward_copy_keeps_output_valid(mocker):
    """Delay copies while independent backward/allocator work runs on compute."""
    shape = (256, 256)
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'empty',
        side_effect=lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
        create=True,
    )
    mocker.patch.object(
        symmetric_p2p.symm_mem, 'rendezvous', side_effect=lambda *a, **k: object(), create=True
    )
    received = [torch.empty(shape, device='cuda') for _ in range(3)]
    transfers = iter(received)
    mocker.patch.object(
        symmetric_p2p.symm_mem,
        'put_signal',
        side_effect=lambda tensor, handle, peer: next(transfers).copy_(tensor),
        create=True,
    )
    group = mocker.Mock()
    group.rank.return_value = 1
    buffers = symmetric_p2p.SymmMemBuffer(shape, torch.float32, group, 1, None)
    previous = None
    independent = torch.ones(shape, device='cuda', requires_grad=True)
    for value in (3, 5, 7):
        output = torch.full(shape, value, device='cuda', dtype=torch.float32)
        with torch.cuda.stream(buffers.send_stream):
            torch.cuda._sleep(20_000_000)
        work = buffers.put_signal(output, dst=0)
        if previous is not None:
            assert previous.copy_done is not work.copy_done
            previous.wait()
        # Like steady 1F1B: submit independent backward before waiting for the
        # current forward's copy. Do not use timing-based overlap assertions.
        independent.square().sum().backward()
        work.wait()
        schedule.deallocate_output_tensor(output, True)
        # Reuse ordinary allocator blocks after the copy barrier, while RMA
        # can still be reading the registered staging buffer on the send stream.
        for _ in range(8):
            scratch = torch.empty(shape, device='cuda')
            scratch.fill_(-1)
            del scratch
        previous = work
    buffers.join()
    torch.cuda.synchronize()
    for tensor, value in zip(received, (3, 5, 7)):
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))
    torch.testing.assert_close(independent.grad, torch.full_like(independent, 6))


@pytest.mark.parametrize('change', ['shape', 'dtype', 'group', 'schedule'])
def test_symmetric_pp_rejects_incompatible_cached_windows(mocker, mock_symmetric_memory, change):
    communicator = _make_symmetric_test_communicator(mocker)
    shape = (2, 1, 4)
    communicator.create_symm_put_wait_buffers(shape, 30, 64)
    if change == 'shape':
        shape = (3, 1, 4)
    elif change == 'dtype':
        communicator.config.pipeline_dtype = torch.bfloat16
    elif change == 'group':
        communicator.pp_group = mocker.Mock()
        communicator.pp_group.size.return_value = 4
    else:
        communicator.config.microbatch_group_size_per_vp_stage = 8
    with pytest.raises(RuntimeError, match='Cached symmetric P2P buffers'):
        communicator.create_symm_put_wait_buffers(shape, 30, 64)


def _symmetric_pp_schedule_dependencies(pp, vp, counts):
    """Reference compute dependencies, independent of the slot-allocation algorithm."""
    orders = []
    for rank in range(pp):
        order = []
        offset = 0
        communicator = SimpleNamespace(
            pp_group=SimpleNamespace(size=lambda: pp, rank=lambda rank=rank: rank),
            virtual_pipeline_model_parallel_size=vp,
        )
        for count in counts:
            table = schedule.get_schedule_table(count, vp, pp)
            _, _, warmup, _ = schedule.get_pp_rank_microbatches(
                count, vp, pp, p2p_communicator=communicator
            )
            iteration_table = [(mb + offset, chunk) for mb, chunk in table]
            order.extend(('F', *iteration_table[i]) for i in range(warmup))
            for i in range(warmup, len(table)):
                mb, chunk = iteration_table[i - warmup]
                order.extend([('F', *iteration_table[i]), ('B', mb, vp - 1 - chunk)])
            order.extend(
                ('B', iteration_table[i][0], vp - 1 - iteration_table[i][1])
                for i in range(len(table) - warmup, len(table))
            )
            offset += count
        orders.append([(rank, *op) for op in order])
    positions = {op: index for order in orders for index, op in enumerate(order)}
    predecessors, followers = {}, {op: [] for op in positions}
    for order in orders:
        for index, op in enumerate(order):
            rank, direction, microbatch, chunk = op
            dependencies = [order[index - 1]] if index else []
            if direction == 'F' and (rank != 0 or chunk != 0):
                dependencies.append(((rank - 1) % pp, 'F', microbatch, chunk - int(rank == 0)))
            if direction == 'B' and (rank != pp - 1 or chunk != vp - 1):
                dependencies.append(((rank + 1) % pp, 'B', microbatch, chunk + int(rank == pp - 1)))
            predecessors[op] = dependencies
            for dependency in dependencies:
                followers[dependency].append(op)
    remaining = {op: len(dependencies) for op, dependencies in predecessors.items()}
    ready = deque(op for op in positions if remaining[op] == 0)
    clocks = {}
    while ready:
        op = ready.popleft()
        clock = [-1] * pp
        for dependency in predecessors[op]:
            clock = [max(a, b) for a, b in zip(clock, clocks[dependency])]
        clock[op[0]] = positions[op]
        clocks[op] = clock
        for follower in followers[op]:
            remaining[follower] -= 1
            if remaining[follower] == 0:
                ready.append(follower)
    assert len(clocks) == len(positions), 'The reference schedule must not deadlock'
    return orders, predecessors, clocks, positions


@pytest.mark.parametrize('pp', [2, 3, 4, 5, 8])
@pytest.mark.parametrize('vp', [2, 3, 4, 8])
@pytest.mark.parametrize('groups', [1, 2, 4, 8])
def test_symmetric_pp_backward_ring_covers_all_rank_skews(pp, vp, groups):
    """Prove reuse follows consumption for every topological execution, not just lockstep."""
    orders, _, clocks, positions = _symmetric_pp_schedule_dependencies(pp, vp, [groups * pp] * 2)
    capacity = 2 * pp - 1
    for rank, order in enumerate(orders):
        last_writer = {}
        for iteration in range(2):
            sends = [
                op
                for op in order
                if op[1] == 'B'
                and (rank != 0 or op[3] != 0)
                and op[2] // (groups * pp) == iteration
            ]
            # reset() restarts the slot indices, but a peer can still be finishing
            # the previous iteration. Check those overwrites as well as ring wraps.
            for index, producer in enumerate(sends):
                slot = index % capacity
                if slot in last_writer:
                    previous = last_writer[slot]
                    consumer = ((rank - 1) % pp, 'B', previous[2], previous[3] - int(rank == 0))
                    assert (
                        clocks[producer][consumer[0]] >= positions[consumer]
                    ), f'Slot reused before consumption: {pp=}, {vp=}, {groups=}, {rank=}, {index=}'
                last_writer[slot] = producer


@pytest.mark.parametrize('pp', [2, 3, 4, 5, 8])
@pytest.mark.parametrize('vp', [2, 3, 4, 8])
@pytest.mark.parametrize('groups', [(1, 1), (1, 4), (4, 1), (4, 16), (16, 4), (16, 16)])
def test_symmetric_pp_forward_slots_cover_all_rank_skews(pp, vp, groups):
    """Every overwrite follows the peer's last reader, also across changing step sizes."""
    counts = [group * pp for group in groups]
    orders, _, clocks, positions = _symmetric_pp_schedule_dependencies(pp, vp, counts)
    plans = [schedule._get_symmetric_forward_slot_plan(pp, vp, count, pp) for count in counts]
    assert plans[0].num_buffers == plans[1].num_buffers
    for rank, order in enumerate(orders):
        last_reader = {}
        offset = 0
        for count, plan in zip(counts, plans):
            producers = [
                op
                for op in order
                if op[1] == 'F'
                and offset <= op[2] < offset + count
                and (rank != pp - 1 or op[3] != vp - 1)
            ]
            assert len(plan.send_slots[rank]) == len(producers)
            for producer, slot in zip(producers, plan.send_slots[rank]):
                assert 0 <= slot < plan.num_buffers
                if slot in last_reader:
                    reader = last_reader[slot]
                    assert clocks[producer][reader[0]] >= positions[reader], (
                        f'Forward overwrite before backward finished: {pp=}, {vp=}, '
                        f'{groups=}, {producer=}, {reader=}, {slot=}'
                    )
                last_reader[slot] = (
                    (rank + 1) % pp,
                    'B',
                    producer[2],
                    producer[3] + int(rank == pp - 1),
                )
            offset += count


@pytest.mark.parametrize(
    'pp,vp,count,group,options',
    [
        (1, 4, 64, 1, {}),
        (4, 1, 64, 4, {}),
        (4, 4, 2, 4, {}),
        (4, 4, 64, 8, {}),
        (4, 4, 65, 4, {}),
        (4, 4, 64, 4, {'forward_only': True}),
        (4, 4, 64, 4, {'overlap_moe_expert_parallel_comm': True}),
        (4, 4, 64, 4, {'delay_wgrad_compute': True}),
    ],
)
def test_symmetric_pp_forward_plan_keeps_other_schedules(pp, vp, count, group, options):
    assert schedule._get_symmetric_forward_slot_plan(pp, vp, count, group, **options) is None
    assert schedule._get_symmetric_recv_slot_plan(pp, vp, count, group, **options) is None


@pytest.mark.parametrize('pp', [3, 4, 5, 8])
@pytest.mark.parametrize('vp', [2, 3, 4, 8])
@pytest.mark.parametrize('groups', [(1, 1), (1, 4), (4, 1), (4, 16), (16, 4), (16, 16)])
def test_symmetric_pp_shared_slots_cover_all_rank_skews(pp, vp, groups):
    """Check both directions against an independent multi-iteration reference DAG.

    Each iteration is planned separately. No barrier between iterations is
    assumed; the final reader of the previous step must precede the next write
    even when the count changes. This also checks forward input/gradient aliasing
    at a backward which needs both at once.
    """
    counts = [group * pp for group in groups]
    _, _, clocks, positions = _symmetric_pp_schedule_dependencies(pp, vp, counts)
    plans = [schedule._get_symmetric_recv_slot_plan(pp, vp, count, pp) for count in counts]
    assert plans[0].num_buffers == plans[1].num_buffers
    messages = [[] for _ in range(pp)]
    offset = 0
    for count, plan in zip(counts, plans):
        assert plan.backward_send_slots is not None
        orders, _, _, _ = _symmetric_pp_schedule_dependencies(pp, vp, [count])
        for rank, order in enumerate(orders):
            for direction, mappings in [('F', plan.send_slots), ('B', plan.backward_send_slots)]:
                writers = [
                    op
                    for op in order
                    if op[1] == direction
                    and not (direction == 'F' and rank == pp - 1 and op[3] == vp - 1)
                    and not (direction == 'B' and rank == 0 and op[3] == 0)
                ]
                assert len(writers) == len(mappings[rank])
                for (_, _, mb, chunk), slot in zip(writers, mappings[rank]):
                    writer = (rank, direction, mb + offset, chunk)
                    if direction == 'F':
                        reader = ((rank + 1) % pp, 'B', mb + offset, chunk + int(rank == pp - 1))
                    else:
                        reader = ((rank - 1) % pp, 'B', mb + offset, chunk - int(rank == 0))
                    assert 0 <= slot < plan.num_buffers
                    messages[reader[0]].append((writer, reader, slot))
        offset += count
    for receiver, incoming in enumerate(messages):
        last_readers = {}
        for writer, reader, slot in sorted(incoming, key=lambda item: positions[item[1]]):
            if slot in last_readers:
                previous = last_readers[slot]
                assert clocks[writer][receiver] >= positions[previous], (
                    f'Shared overwrite before consumption: {pp=}, {vp=}, {groups=}, '
                    f'{writer=}, {previous=}, {slot=}'
                )
            last_readers[slot] = reader


def test_symmetric_pp_two_ranks_keep_separate_signal_peers():
    plan = schedule._get_symmetric_recv_slot_plan(2, 4, 64, 2)
    assert plan == schedule._get_symmetric_forward_slot_plan(2, 4, 64, 2)
    assert plan.backward_send_slots is None


@pytest.mark.parametrize('initially_shared', [False, True])
def test_symmetric_pp_rejects_changing_pool_mode(mocker, mock_symmetric_memory, initially_shared):
    communicator = _make_symmetric_test_communicator(mocker)
    shared = schedule._get_symmetric_recv_slot_plan(4, 4, 64, 4)
    separate = schedule._get_symmetric_forward_slot_plan(4, 4, 64, 4)
    initial, changed = (shared, separate) if initially_shared else (separate, shared)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64, recv_slot_plan=initial)
    forward = communicator.symm_buffers['send_next_recv_prev']
    backward = communicator.symm_buffers['send_prev_recv_next']
    original_plans = forward.send_slots, backward.send_slots
    with pytest.raises(RuntimeError, match='Cached symmetric P2P buffers'):
        communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 64, recv_slot_plan=changed)
    assert (forward.send_slots, backward.send_slots) == original_plans


@pytest.mark.parametrize('change', ['shape', 'dtype', 'group', 'capacity'])
def test_symmetric_pp_rejects_incompatible_shared_pool(mocker, mock_symmetric_memory, change):
    shape, dtype, group = (2, 1, 4), torch.float32, mocker.Mock()
    source = symmetric_p2p.SymmMemBuffer(shape, dtype, group, 2, None)
    allocations = symmetric_p2p.symm_mem.empty.call_count
    if change == 'shape':
        shape = (3, 1, 4)
    elif change == 'dtype':
        dtype = torch.bfloat16
    elif change == 'group':
        group = mocker.Mock()
    with pytest.raises(ValueError, match='Shared symmetric P2P'):
        if change == 'capacity':
            symmetric_p2p.SymmMemBuffer(
                shape, dtype, group, 3, None, shared_recv_buffers=source.recv_buffers
            )
        else:
            symmetric_p2p.SymmWaitBuffer(
                shape, dtype, group, torch.cuda.Stream(), shared_buffer=source.recv_buffers[0]
            )
    assert symmetric_p2p.symm_mem.empty.call_count == allocations
    assert symmetric_p2p.symm_mem.rendezvous.call_count == allocations


def test_symmetric_pp_planned_slot_selection_and_guards(mocker, mock_symmetric_memory):
    communicator = _make_symmetric_test_communicator(mocker)
    plan = schedule._get_symmetric_forward_slot_plan(4, 4, 4, 4)
    communicator.create_symm_put_wait_buffers((2, 1, 4), 30, 4, recv_slot_plan=plan)
    forward = communicator.symm_buffers['send_next_recv_prev']
    assert forward.send_slots == plan.send_slots[1]
    assert forward.recv_slots == plan.send_slots[0]

    # Exercise a non-FIFO order through all three production lookup sites.
    forward.send_slots = (2, 0, 2, 1)
    forward.recv_slots = (1, 2, 0, 2)
    for send_slot, recv_slot in zip(forward.send_slots, forward.recv_slots):
        tensor = forward.get_recv_buffer()
        assert tensor.data_ptr() == forward.recv_buffers[recv_slot].buffer.data_ptr()
        request = forward.wait_signal(src=0, tensor=tensor)
        assert request is forward.recv_buffers[recv_slot]
        forward.put_signal(tensor, dst=2)
        assert mock_symmetric_memory.trace[-1][1][1] is forward.hdls[send_slot]
    with pytest.raises(RuntimeError, match='exceeded the planned'):
        forward.get_recv_buffer()
    with pytest.raises(RuntimeError, match='exceeded the planned'):
        forward.wait_signal(src=0)
    with pytest.raises(RuntimeError, match='exceeded the planned'):
        forward.put_signal(tensor, dst=2)

    forward.reset(send_slots=(0,), recv_slots=(0,))
    with pytest.raises(RuntimeError, match='planned sends completed'):
        forward.reset()
    forward.put_signal(tensor, dst=2)
    with pytest.raises(RuntimeError, match='planned receives completed'):
        forward.reset()
    tensor = forward.get_recv_buffer()
    with pytest.raises(RuntimeError, match='does not match its planned slot'):
        forward.wait_signal(src=0, tensor=forward.recv_buffers[1].buffer)
    forward.wait_signal(src=0, tensor=tensor)
    with pytest.raises(ValueError, match='registered buffer capacity'):
        forward.reset(send_slots=(forward.num_recv_buffers,))

    windows = tuple(buffer.buffer.data_ptr() for buffer in forward.recv_buffers)
    for count in (64, 8):
        next_plan = schedule._get_symmetric_forward_slot_plan(4, 4, count, 4)
        communicator.create_symm_put_wait_buffers((2, 1, 4), 30, count, recv_slot_plan=next_plan)
        assert tuple(buffer.buffer.data_ptr() for buffer in forward.recv_buffers) == windows
        # Complete this synthetic communication-only step before changing its plan.
        for _ in forward.send_slots:
            forward.put_signal(tensor, dst=2)
        for _ in forward.recv_slots:
            tensor = forward.get_recv_buffer()
            forward.wait_signal(src=0, tensor=tensor)


@pytest.mark.parametrize('new_communicator', [False, True])
@pytest.mark.parametrize('shared_pool', [False, True])
def test_symmetric_pp_evaluation_preserves_planned_windows(
    mocker, mock_symmetric_memory, new_communicator, shared_pool
):
    communicator = _make_symmetric_test_communicator(mocker)
    shape = (2, 1, 4)
    # Evaluation before training must not allocate the larger legacy ring either.
    communicator.create_symm_put_wait_buffers(shape, 30, 4, forward_only=True)
    assert not communicator._use_symmetric_memory_p2p
    assert not communicator.symm_buffers

    planner = (
        schedule._get_symmetric_recv_slot_plan
        if shared_pool
        else schedule._get_symmetric_forward_slot_plan
    )
    plan = planner(4, 4, 4, 4)
    communicator.create_symm_put_wait_buffers(shape, 30, 4, recv_slot_plan=plan)
    forward = communicator.symm_buffers['send_next_recv_prev']
    pointers = [buffer.buffer.data_ptr() for buffer in forward.recv_buffers]
    for _ in forward.send_slots:
        forward.put_signal(torch.ones(shape), dst=2)
    for _ in forward.recv_slots:
        forward.wait_signal(src=0, tensor=forward.get_recv_buffer())
    backward = communicator.symm_buffers['send_prev_recv_next']
    if shared_pool:
        for _ in backward.send_slots:
            backward.put_signal(torch.ones(shape), dst=0)
        for _ in backward.recv_slots:
            backward.wait_signal(src=2, tensor=backward.get_recv_buffer())
    communicator.symm_join()

    if new_communicator:
        group = communicator.pp_group
        communicator = _make_symmetric_test_communicator(mocker)
        communicator.pp_group = group
    communicator.next_rank, communicator.prev_rank = 13, 5
    communicator.next_rank_pg, communicator.prev_rank_pg = 2, 0
    communicator.config.batch_p2p_comm = False
    ordinary_p2p = mocker.patch.object(p2p, '_p2p_ops', return_value={}, create=True)
    # An evaluation sequence length may differ without touching registered windows.
    eval_shape = (3, 1, 4)
    communicator.create_symm_put_wait_buffers(eval_shape, 30, 8, forward_only=True)
    received, _, _ = communicator._communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=eval_shape,
    )
    assert tuple(received.shape) == eval_shape
    assert received.data_ptr() not in pointers
    assert ordinary_p2p.call_args.kwargs['next_pipeline_rank'] == 13
    assert ordinary_p2p.call_args.kwargs['prev_pipeline_rank'] == 5
    assert communicator.config.use_symmetric_memory_p2p
    communicator.prepare_input_tensor_grad(received)
    assert received.grad is None
    communicator.symm_join()

    next_plan = planner(4, 4, 64, 4)
    communicator.create_symm_put_wait_buffers(shape, 30, 64, recv_slot_plan=next_plan)
    assert communicator._use_symmetric_memory_p2p
    assert communicator.symm_buffers['send_next_recv_prev'] is forward
    assert forward.send_slots == next_plan.send_slots[1]
    assert [buffer.buffer.data_ptr() for buffer in forward.recv_buffers] == pointers
    assert backward.shared_recv_buffers == shared_pool
    if shared_pool:
        assert backward.send_slots == next_plan.backward_send_slots[1]
        assert backward.recv_slots == next_plan.backward_send_slots[2]
        assert [buffer.buffer.data_ptr() for buffer in backward.recv_buffers] == pointers


@pytest.mark.parametrize('reverse_rank_priority', [False, True])
@pytest.mark.parametrize('shared_pool', [False, True])
def test_symmetric_pp_planned_slots_match_pipeline_gradients(
    mocker, mock_symmetric_memory, reverse_rank_priority, shared_pool
):
    """Real CPU autograd through PP4/VPP4 with skew, slot reuse and varying step sizes.

    Only CUDA streams and transport are mocked. Both directions use the actual
    SymmMemBuffer methods, including each transient input gradient's staging copy.
    """
    pp, vp, shape = 4, 4, (2, 1, 4)
    counts = [4, 64, 8]
    planner = (
        schedule._get_symmetric_recv_slot_plan
        if shared_pool
        else schedule._get_symmetric_forward_slot_plan
    )
    plans = [planner(pp, vp, count, pp) for count in counts]
    buffers = []
    for rank in range(pp):
        group = mocker.Mock()
        group.rank.return_value = rank
        rank_buffers = {}
        backward_capacity = plans[0].num_buffers if shared_pool else 2 * pp - 1
        for direction, capacity in [('F', plans[0].num_buffers), ('B', backward_capacity)]:
            buffer = symmetric_p2p.SymmMemBuffer(
                shape,
                torch.float32,
                group,
                capacity,
                None,
                requires_grad=direction == 'F',
                shared_recv_buffers=(
                    rank_buffers['F'].recv_buffers if shared_pool and direction == 'B' else None
                ),
            )
            if not shared_pool or direction == 'F':
                for slot, receive in enumerate(buffer.recv_buffers):
                    receive.hdl = SimpleNamespace(rank=rank, slot=slot)
            buffer.hdls = [receive.hdl for receive in buffer.recv_buffers]
            rank_buffers[direction] = buffer
        buffers.append(rank_buffers)

    pending = set()

    def put(tensor, handle, dst):
        direction = 'F' if dst == (handle.rank + 1) % pp else 'B'
        target = buffers[dst][direction].recv_buffers[handle.slot].buffer
        key = (dst, target.data_ptr())
        assert key not in pending, 'Overwrote an unreceived message'
        target.copy_(tensor)
        pending.add(key)

    def wait(handle, src):
        direction = 'F' if src == (handle.rank - 1) % pp else 'B'
        target = buffers[handle.rank][direction].recv_buffers[handle.slot].buffer
        pending.remove((handle.rank, target.data_ptr()))

    mocker.patch.object(symmetric_p2p.symm_mem, 'put_signal', side_effect=put)
    mocker.patch.object(symmetric_p2p.symm_mem, 'wait_signal', side_effect=wait)
    weights = [torch.tensor(1 + stage / 100, requires_grad=True) for stage in range(pp * vp)]
    reference_weights = [weight.detach().clone().requires_grad_() for weight in weights]
    activations, outputs, input_grads = {}, {}, {}
    orders, predecessors, _, _ = _symmetric_pp_schedule_dependencies(pp, vp, counts)
    progress, step = [0] * pp, [-1] * pp
    completed = set()
    rank_order = list(reversed(range(pp))) if reverse_rank_priority else list(range(pp))
    while any(progress[rank] < len(orders[rank]) for rank in range(pp)):
        rank = next(
            rank
            for rank in rank_order
            if progress[rank] < len(orders[rank])
            and all(dep in completed for dep in predecessors[orders[rank][progress[rank]]])
        )
        op = orders[rank][progress[rank]]
        _, direction, microbatch, chunk = op
        iteration = 0 if microbatch < counts[0] else (1 if microbatch < sum(counts[:2]) else 2)
        forward, backward = buffers[rank]['F'], buffers[rank]['B']
        if step[rank] != iteration:
            plan = plans[iteration]
            forward.reset(
                send_slots=plan.send_slots[rank], recv_slots=plan.send_slots[(rank - 1) % pp]
            )
            backward.reset(
                send_slots=plan.backward_send_slots[rank] if shared_pool else None,
                recv_slots=plan.backward_send_slots[(rank + 1) % pp] if shared_pool else None,
            )
            step[rank] = iteration
        key = (rank, microbatch, chunk)
        if direction == 'F':
            if rank == 0 and chunk == 0:
                tensor = torch.full(shape, 1 + microbatch / 100, requires_grad=True)
            else:
                tensor = forward.get_recv_buffer()
                forward.wait_signal((rank - 1) % pp, tensor).wait()
            output = tensor * weights[chunk * pp + rank] + 0.1
            activations[key], outputs[key] = tensor, output
            if rank != pp - 1 or chunk != vp - 1:
                forward.put_signal(output, (rank + 1) % pp).wait()
        else:
            tensor, output = activations.pop(key), outputs.pop(key)
            if rank == pp - 1 and chunk == vp - 1:
                grad = torch.ones_like(output)
            else:
                grad = backward.get_recv_buffer()
                backward.wait_signal((rank + 1) % pp, grad).wait()
            for send in backward.send_buffers:
                send.wait()
            forward.prepare_grad(tensor)
            output.backward(grad)
            input_grad = schedule._take_input_tensor_grads([tensor])[0]
            if rank != 0 or chunk != 0:
                assert tensor.grad is None
                backward.put_signal(input_grad, (rank - 1) % pp).wait()
            else:
                input_grads[microbatch] = input_grad.clone()
        completed.add(op)
        progress[rank] += 1

    assert not pending and not activations and not outputs
    for microbatch in range(sum(counts)):
        reference_input = torch.full(shape, 1 + microbatch / 100, requires_grad=True)
        output = reference_input
        for weight in reference_weights:
            output = output * weight + 0.1
        output.sum().backward()
        torch.testing.assert_close(input_grads[microbatch], reference_input.grad)
    for weight, reference in zip(weights, reference_weights):
        torch.testing.assert_close(weight.grad, reference.grad)
    for rank_buffers in buffers:
        assert all(
            buffer.buffer.grad is None
            for direction in ('F', 'B')
            for buffer in rank_buffers[direction].recv_buffers
        )


@pytest.mark.parametrize('mtp_standalone', [False, True])
def test_dynamic_shapes_rejected_before_eager_initialization(mocker, pp_group, mtp_standalone):
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.bfloat16,
        use_symmetric_memory_p2p=True,
        variable_seq_lengths=not mtp_standalone,
        mtp_standalone=mtp_standalone,
    )
    current_device = mocker.patch.object(torch.cuda, 'current_device')
    set_backend = mocker.patch.object(p2p.symm_mem, 'set_backend')
    get_mem_pool = mocker.patch.object(p2p.symm_mem, 'get_mem_pool')

    match = 'standalone MTP' if mtp_standalone else 'variable sequence lengths'
    with pytest.raises(AssertionError, match=match):
        p2p.P2PCommunicator(pp_group, config)

    current_device.assert_not_called()
    pp_group._get_backend.assert_not_called()
    set_backend.assert_not_called()
    get_mem_pool.assert_not_called()


def test_fixed_packed_symmetric_p2p_skips_shape_exchange(mocker, pp_group):
    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.bfloat16,
        use_symmetric_memory_p2p=True,
        variable_seq_lengths=True,
        pipeline_p2p_fixed_shape=True,
        sequence_packing_scheduler='dp_balanced',
        max_seqlen_per_dp_cp_rank=4096,
        pad_packed_seq_alignment='max',
    )
    mocker.patch.object(torch.cuda, 'current_device', return_value=3)
    mocker.patch.object(p2p.symm_mem, 'set_backend')
    mocker.patch.object(p2p.symm_mem, 'get_mem_pool')
    communicator = p2p.P2PCommunicator(pp_group, config)
    exchange_shapes = mocker.patch.object(communicator, '_communicate_shapes')
    buffers = {'send_next_recv_prev': mocker.Mock(), 'send_prev_recv_next': mocker.Mock()}
    mocker.patch.object(p2p.P2PCommunicator, 'symm_buffers', buffers)
    p2p_ops = mocker.patch.object(p2p, '_symm_mem_p2p_ops', return_value={})

    recv_prev, recv_next, _ = communicator._communicate(
        tensor_send_next=mocker.sentinel.send_next,
        tensor_send_prev=mocker.sentinel.send_prev,
        recv_prev=True,
        recv_next=True,
        tensor_shape=(4096, 1, 128),
    )

    exchange_shapes.assert_not_called()
    assert recv_prev is buffers['send_next_recv_prev'].get_recv_buffer.return_value
    assert recv_next is buffers['send_prev_recv_next'].get_recv_buffer.return_value
    p2p_ops.assert_called_once_with(
        tensor_send_prev=mocker.sentinel.send_prev,
        tensor_recv_prev=recv_prev,
        tensor_send_next=mocker.sentinel.send_next,
        tensor_recv_next=recv_next,
        symm_buffers=buffers,
        group=pp_group,
        prev_pipeline_rank=0,
        next_pipeline_rank=2,
    )


@pytest.mark.parametrize(
    'seq_length,cp_size,tp_size,sequence_parallel,packed_capacity,expected',
    [
        (65536, 16, 1, False, None, 4096),
        (4096, 1, 1, False, None, 4096),
        (65536, 16, 2, True, None, 2048),
        (65536, 16, 1, False, 6144, 6144),
        (4096, 1, 1, False, 6144, 6144),
        (65536, 16, 2, True, 6144, 3072),
    ],
)
def test_fixed_p2p_sequence_length(
    mocker, seq_length, cp_size, tp_size, sequence_parallel, packed_capacity, expected
):
    config = SimpleNamespace(
        variable_seq_lengths=packed_capacity is not None,
        pipeline_p2p_fixed_shape=packed_capacity is not None,
        max_seqlen_per_dp_cp_rank=packed_capacity,
        sequence_parallel=sequence_parallel,
        hidden_size=128,
    )
    tp_group, cp_group = mocker.Mock(), mocker.Mock()
    tp_group.size.return_value = tp_size
    cp_group.size.return_value = cp_size

    # Interleaved PP uses this helper when preallocating the symmetric buffers.
    assert schedule._get_p2p_seq_length(seq_length, config, tp_group, cp_group) == expected
    assert schedule.get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=1,
        decoder_seq_length=None,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    ) == [(expected, 1, 128)]
    if packed_capacity is not None:
        cp_group.size.assert_not_called()


def test_get_forward_backward_func():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()


def test_deallocate_output_tensor():
    out = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schedule.deallocate_output_tensor(out)
    assert out.nelement() == 6


@contextmanager
def _no_sync():
    yield


def _patch_hybrid_cp_parallel_state(monkeypatch, *, is_first_tp_rank):
    monkeypatch.setattr(
        hybrid_cp_schedule.parallel_state,
        "get_data_parallel_rank",
        lambda with_context_parallel=False: 0,
    )
    monkeypatch.setattr(
        hybrid_cp_schedule.parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 0 if is_first_tp_rank else 1,
    )
    monkeypatch.setattr(
        hybrid_cp_schedule.parallel_state, "get_tensor_model_parallel_src_rank", lambda: 0
    )
    monkeypatch.setattr(
        hybrid_cp_schedule.parallel_state, "get_tensor_model_parallel_group", lambda: "tp_group"
    )
    monkeypatch.setattr(
        hybrid_cp_schedule.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=False: "dp_cp_group",
    )


def _patch_hybrid_cp_cpu_tensors(monkeypatch):
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(hybrid_cp_schedule.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(
        hybrid_cp_schedule.torch.cuda, "current_device", lambda: torch.device("cpu")
    )


def test_hybrid_context_parallel_forward_backward_passes_local_cp_size(monkeypatch):
    _patch_hybrid_cp_cpu_tensors(monkeypatch)
    _patch_hybrid_cp_parallel_state(monkeypatch, is_first_tp_rank=True)

    monkeypatch.setattr(
        hybrid_cp_schedule.torch.distributed, "broadcast", lambda *args, **kwargs: None
    )
    barrier_groups = []
    monkeypatch.setattr(
        hybrid_cp_schedule.torch.distributed,
        "barrier",
        lambda group=None: barrier_groups.append(group),
    )

    batch = [{"id": 0}, {"id": 1}, {"id": 2}]
    sample_id_groups = [[[0], [0], []], [[1, 2], [1], [1, 2]]]
    forward_calls = []

    def fake_forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        cp_group_size,
        **kwargs,
    ):
        assert isinstance(data_iterator, RerunDataIterator)
        sample = next(data_iterator)
        forward_calls.append(
            {
                "sample_id": sample["id"],
                "local_cp_size": int(sample["local_cp_size"].item()),
                "local_cp_size_dtype": sample["local_cp_size"].dtype,
                "cp_group_size": cp_group_size,
                "current_microbatch": kwargs["current_microbatch"],
                "is_first_microbatch": kwargs["is_first_microbatch"],
            }
        )
        return torch.tensor(float(kwargs["current_microbatch"])), torch.tensor(10)

    backward_calls = []

    def fake_backward_step(input_tensor, output_tensor, output_tensor_grad, config):
        backward_calls.append((input_tensor, output_tensor.item(), output_tensor_grad, config))

    monkeypatch.setattr(schedule, "forward_step", fake_forward_step)
    monkeypatch.setattr(schedule, "backward_step", fake_backward_step)

    config = SimpleNamespace()
    forward_data_store, total_num_tokens = (
        hybrid_cp_schedule.hybrid_context_parallel_forward_backward(
            forward_step_func=None,
            data_iterator=iter([(batch, sample_id_groups)]),
            model="model",
            num_microbatches=3,
            input_tensor="input",
            output_tensor_grad="grad",
            forward_data_store=[],
            config=config,
            collect_non_loss_data=False,
            first_val_step=True,
            forward_only=False,
            no_sync_func=_no_sync,
            total_num_tokens=0,
            check_first_val_step=lambda first_val_step, forward_only, is_first: is_first,
            model_type="unused",
        )
    )

    assert forward_data_store == []
    assert total_num_tokens == 30
    assert forward_calls == [
        {
            "sample_id": 0,
            "local_cp_size": 2,
            "local_cp_size_dtype": torch.int32,
            "cp_group_size": 2,
            "current_microbatch": 0,
            "is_first_microbatch": True,
        },
        {
            "sample_id": 1,
            "local_cp_size": 3,
            "local_cp_size_dtype": torch.int32,
            "cp_group_size": 3,
            "current_microbatch": 1,
            "is_first_microbatch": False,
        },
        {
            "sample_id": 2,
            "local_cp_size": 2,
            "local_cp_size_dtype": torch.int32,
            "cp_group_size": 2,
            "current_microbatch": 2,
            "is_first_microbatch": False,
        },
    ]
    assert [(call[0], call[1], call[2]) for call in backward_calls] == [
        ("input", 0.0, "grad"),
        ("input", 1.0, "grad"),
        ("input", 2.0, "grad"),
    ]
    assert all(call[3] is config for call in backward_calls)
    assert "dp_cp_group" in barrier_groups


def test_hybrid_context_parallel_non_first_tp_rank_uses_broadcast_cp_size(monkeypatch):
    _patch_hybrid_cp_parallel_state(monkeypatch, is_first_tp_rank=False)
    monkeypatch.setattr(
        hybrid_cp_schedule.torch.cuda, "current_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(hybrid_cp_schedule.torch.distributed, "barrier", lambda group=None: None)

    broadcast_values = [
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
    ]

    def fake_broadcast(item, src, group=None):
        item.copy_(broadcast_values.pop(0))

    monkeypatch.setattr(hybrid_cp_schedule.torch.distributed, "broadcast", fake_broadcast)

    forward_calls = []

    def fake_forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        cp_group_size,
        **kwargs,
    ):
        forward_calls.append((data_iterator, cp_group_size, kwargs["current_microbatch"]))
        return torch.tensor(0.0), torch.tensor(4)

    monkeypatch.setattr(schedule, "forward_step", fake_forward_step)
    monkeypatch.setattr(
        schedule,
        "backward_step",
        lambda input_tensor, output_tensor, output_tensor_grad, config: None,
    )

    _, total_num_tokens = hybrid_cp_schedule.hybrid_context_parallel_forward_backward(
        forward_step_func=None,
        data_iterator=None,
        model="model",
        num_microbatches=1,
        input_tensor="input",
        output_tensor_grad="grad",
        forward_data_store=[],
        config=SimpleNamespace(),
        collect_non_loss_data=False,
        first_val_step=True,
        forward_only=True,
        no_sync_func=_no_sync,
        total_num_tokens=0,
        check_first_val_step=lambda first_val_step, forward_only, is_first: is_first,
        model_type="unused",
    )

    assert forward_calls == [(None, 7, 0)]
    assert total_num_tokens == 4
    assert broadcast_values == []


@pytest.mark.parametrize("calculate_per_token_loss,expected_scale", [(False, 6.0), (True, 3.0)])
def test_dsa_indexer_loss_scale_matches_schedule_cp_scaling(
    calculate_per_token_loss, expected_scale
):
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    config = SimpleNamespace(
        calculate_per_token_loss=calculate_per_token_loss,
        experimental_attention_variant_loss_scale_func=DSAIndexerLossAutoScaler.set_loss_scale,
        experimental_attention_variant='dsa',
        grad_scale_func=lambda tensor: tensor * 3.0,
        num_moe_experts=None,
        mtp_num_layers=None,
        timers=None,
    )
    forward_data_store = []

    def loss_func(output_tensor):
        return output_tensor.clone(), torch.tensor(4), {'loss_reduced': output_tensor.detach()}

    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    schedule.forward_step_calc_loss(
        model=None,
        output_tensor=torch.tensor(8.0),
        loss_func=loss_func,
        config=config,
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=2,
        forward_data_store=forward_data_store,
        cp_group_size=4,
        is_last_stage=True,
    )

    torch.testing.assert_close(
        DSAIndexerLossAutoScaler.main_loss_backward_scale, torch.tensor([expected_scale])
    )


def test_dsa_indexer_loss_scale_accepts_dict_output_tensor():
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    config = SimpleNamespace(
        calculate_per_token_loss=True,
        experimental_attention_variant_loss_scale_func=DSAIndexerLossAutoScaler.set_loss_scale,
        experimental_attention_variant='dsa',
        grad_scale_func=lambda tensor: tensor * 5.0,
        num_moe_experts=None,
        mtp_num_layers=None,
        timers=None,
    )

    forward_data_store = []

    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    schedule.forward_step_calc_loss(
        model=None,
        output_tensor={'loss': torch.tensor(8.0)},
        loss_func=None,
        config=config,
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=2,
        forward_data_store=forward_data_store,
        cp_group_size=4,
        is_last_stage=True,
    )

    assert len(forward_data_store) == 1
    torch.testing.assert_close(forward_data_store[0]['loss'], torch.tensor(8.0))
    torch.testing.assert_close(
        DSAIndexerLossAutoScaler.main_loss_backward_scale, torch.tensor([5.0])
    )


def test_dsa_indexer_loss_scale_defaults_from_variant_without_mutating_config():
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    config = SimpleNamespace(
        calculate_per_token_loss=True,
        experimental_attention_variant_loss_scale_func=None,
        experimental_attention_variant='dsa',
        grad_scale_func=lambda tensor: tensor * 7.0,
        num_moe_experts=None,
        mtp_num_layers=None,
        timers=None,
    )

    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    schedule.forward_step_calc_loss(
        model=None,
        output_tensor=torch.tensor(8.0),
        loss_func=None,
        config=config,
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=2,
        forward_data_store=[],
        cp_group_size=4,
        is_last_stage=True,
    )

    assert config.experimental_attention_variant_loss_scale_func is None
    torch.testing.assert_close(
        DSAIndexerLossAutoScaler.main_loss_backward_scale, torch.tensor([7.0])
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "pipeline_model_parallel_size,microbatch_group_size_per_vp_stage",
    [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("num_microbatches", [8, 32])
@pytest.mark.parametrize("virtual_pipeline_model_parallel_size", [None, 2, 4, 8])
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
    if pipeline_model_parallel_size == 1 and virtual_pipeline_model_parallel_size is not None:
        return

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    num_model_chunks = (
        virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is not None
        else 1
    )

    _, _, num_warmup_microbatches, _ = schedule.get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, False
    )
    schedule_table = schedule.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )

    assert max(order) == num_model_chunks
    assert len(order) == num_microbatches * num_model_chunks * 2
    order_cnt = {}
    accumulated_order = 0
    for o in order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0
    assert 0 not in order_cnt
    for k, v in order_cnt.items():
        assert -k in order_cnt and order_cnt[-k] == v

    layers_per_chunk = 2
    num_layers_per_chunk = [layers_per_chunk] * num_model_chunks
    # disable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, False
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk
    assert len(chunk_id_list) == len(overlapped_order)
    order_cnt = {}
    accumulated_order = 0
    for o in overlapped_order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0

    # enable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, True
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk * 3 // 2
    assert len(chunk_id_list) == len(overlapped_order)
    from math import ceil

    order_cnt = {}
    accumulated_order = 0
    prev_o = 0
    for o in overlapped_order:
        if ceil(o) != o:
            assert prev_o - 0.5 == o
        else:
            order_cnt[o] = order_cnt.get(o, 0) + 1
            if o < 0:
                assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
            elif -o in order_cnt:
                assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        prev_o = o
    assert accumulated_order < 0

    Utils.destroy_model_parallel()


def test_forward_backward_func_without_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_forward_backward_func_with_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=[model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]
    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_uneven_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(8, 1)
    model_a.vp_stage = 0
    model_b.vp_stage = 1

    def set_input_tensor(input_tensor):
        return None

    model_a.set_input_tensor = set_input_tensor
    model_b.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model_a.config = config
    model_b.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_without_interleaving produces the same output
    with and without explicit process group parameters."""

    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    # Create model
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    # Mock custom_backward to avoid actual computation
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    # Common arguments for both calls
    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': None,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    # First call: without providing process group parameters (they'll be created internally)
    losses_reduced_default = schedule.forward_backward_pipelining_without_interleaving(
        **common_args
    )

    grid = HyperCommGrid([2, 1, 4, 1], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = dp_cp_group
    pg_collection.cp = grid.create_pg("cp")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=p2p_communicator, pg_collection=pg_collection, **common_args
    )

    assert len(losses_reduced_default) == len(
        losses_reduced_explicit
    ), "Output lengths should be identical"

    for i, (default_loss, explicit_loss) in enumerate(
        zip(losses_reduced_default, losses_reduced_explicit)
    ):
        assert (
            default_loss == explicit_loss
        ), f"Loss at index {i} should be identical between default and explicit PG calls"
    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_with_interleaving produces the same output
    with and without explicit process group parameters."""

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.cp = grid.create_pg("cp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = grid.create_pg(["dp", "cp"])

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_no_pipelining_with_custom_pgs(mocker):
    """Validate no-pipeline schedule when explicit custom PGs are provided."""

    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank_local = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank_local, {'loss_reduced': rank_local}

        dummy_inp = torch.ones(1, 4)
        return model(dummy_inp), loss_func

    # Simple model.
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'
    model.set_input_tensor = lambda _tensor: None  # type: ignore[assignment]

    # Minimal config.
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    grid = HyperCommGrid([2, 1, 1, 4], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.pp = pp_group
    pg_collection.dp_cp = dp_cp_group

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 10),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
        pg_collection=pg_collection,
    )

    expected = {'loss_reduced': Utils.rank}
    for l in losses_reduced:
        assert l['loss_reduced'] == expected['loss_reduced']

    Utils.destroy_model_parallel()
