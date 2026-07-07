"""Wave 38.5: V2CompatBuffer - V1 Buffer surface wrapping V2 ElasticBuffer.

SGLang PR #24443 added V2 construction (_build_v2_buffer) but the dispatcher
in deepep.py:_dispatch_core still uses V1-era methods (get_dispatch_layout,
buffer.dispatch with layout kwargs, buffer.combine with `handle` positional).

ElasticBuffer's actual surface is layout-less; it derives dispatch layout
internally from num_max_tokens_per_rank, num_experts, num_topk.

This wrapper bridges the two: each method accepts the V1 signature, calls
into ElasticBuffer with the V2 signature, and converts return tuples back
to the V1 shape so SGLang's _dispatch_core doesn't need to change.

Activation: SGLANG_DEEPEP_USE_V2=1 + _build_v2_buffer returns a
V2CompatBuffer wrapping ElasticBuffer (instead of raw ElasticBuffer).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from deep_ep import ElasticBuffer


class V2CompatBuffer:
    """V1 Buffer-shaped wrapper over deep_ep.ElasticBuffer."""

    def __init__(
        self,
        elastic: ElasticBuffer,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int = 8,
    ):
        self._elastic = elastic
        self._num_experts = num_experts
        self._num_max_tokens_per_rank = num_max_tokens_per_rank
        self._num_topk = num_topk
        self._last_handle = None

    @property
    def group_size(self) -> int:
        return self._elastic.group_size

    @property
    def rank(self) -> int:
        return self._elastic.rank

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Any = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[None, None, None, None, Any]:
        """V1 layout probe. ElasticBuffer derives layout internally.

        Returns a 5-tuple shape-compatible with V1:
        (num_tokens_per_rank, num_tokens_per_rdma_rank,
         num_tokens_per_expert, is_token_in_rank, previous_event)

        All layout slots are None because ElasticBuffer ignores them and
        re-derives from num_max_tokens_per_rank + num_experts + topk.
        """
        return (None, None, None, None, previous_event)

    def dispatch(
        self,
        x,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        previous_event=None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        expert_alignment: int = 1,
        config=None,
    ):
        """V1 signature -> ElasticBuffer.dispatch.

        Returns V1 6-tuple:
        (recv_x, recv_topk_ids, recv_topk_weights,
         num_recv_tokens_per_expert, handle, event)
        """
        # Wave 38.7: match the PROVEN Megatron PR #4632 V2 call pattern
        # exactly (dmvevents/Megatron-LM@2f149cf fused_a2a.py:214-228).
        # num_sms=0 lets V2 use handle.num_sms (avoids CUDA 719 at
        # csrc/jit/handle.hpp:86). Pass num_experts only when the shim has
        # a positive value; 0/None lets V2 auto-derive.
        dispatch_kwargs = dict(
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=self._num_max_tokens_per_rank,
            num_sms=0,
            num_qps=0,
            expert_alignment=expert_alignment,
            previous_event=previous_event,
            async_with_compute_stream=False,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_expand=False,
        )
        if self._num_experts and self._num_experts > 0:
            dispatch_kwargs["num_experts"] = self._num_experts
        out_x, out_topk_idx, out_topk_weights, handle, event = self._elastic.dispatch(
            x, **dispatch_kwargs
        )
        self._last_handle = handle
        num_recv_tokens_per_expert = getattr(
            handle, "num_recv_tokens_per_expert", None
        )
        return (
            out_x,
            out_topk_idx,
            out_topk_weights,
            num_recv_tokens_per_expert,
            handle,
            event,
        )

    def combine(
        self,
        x,
        handle=None,
        topk_weights=None,
        bias=None,
        async_finish: bool = False,
        previous_event=None,
        allocate_on_comm_stream: bool = False,
        config=None,
    ):
        """V1 combine signature -> ElasticBuffer.combine.

        V1 returns (combined_x, combined_topk_weights, event).
        ElasticBuffer.combine returns (x, topk_weights, event).
        """
        # Wave 38.7: mirror Megatron PR #4632 fused_a2a.py:379-387.
        # num_sms=0 / num_qps=0 let V2 pull from handle (captured at dispatch).
        use_handle = handle if handle is not None else self._last_handle
        return self._elastic.combine(
            x,
            handle=use_handle,
            topk_weights=topk_weights,
            bias=bias,
            num_sms=0,
            num_qps=0,
            previous_event=previous_event,
            async_with_compute_stream=False,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

    @staticmethod
    def get_dispatch_config(group_size: int):
        return None

    @staticmethod
    def get_combine_config(group_size: int):
        return None

    @staticmethod
    def capture():
        """V1 `Buffer.capture()` returned a torch CUDA event wrapper."""
        return None

    def __getattr__(self, name):
        return getattr(self._elastic, name)
