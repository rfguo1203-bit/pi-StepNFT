# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process embodied training entry for pdb debugging.

This entry keeps the same embodied computation path as train_embodied_agent.py
for single-machine/single-GPU runs, while removing Ray and using local channels.
"""

import copy
import gc
import json
import os
import socket
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_fsdp_cfg
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.env_manager import EnvManager
from rlinf.models import get_model
from rlinf.scheduler import Channel
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    compute_split_num,
    print_metrics_table,
)
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.utils import get_model_weights_id
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _init_single_process_dist():
    if dist.is_initialized():
        yield
        return

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("NODE_RANK", "0")

    if has_cuda:
        torch.cuda.set_device(0)

    dist.init_process_group(
        backend=backend,
        rank=0,
        world_size=1,
        timeout=timedelta(minutes=30),
    )

    try:
        yield
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _setup_local_worker_attrs(worker: Any, group_name: str) -> None:
    worker._rank = 0
    worker._world_size = 1
    worker._group_name = group_name
    worker._cluster_node_rank = 0
    worker._local_accelerator_rank = 0
    worker._node_local_rank = 0
    worker._node_local_world_size = 1
    worker._local_rank = 0
    worker._local_world_size = 1
    worker._is_ray_actor = False
    worker._timer_metrics = {}
    worker._logger = get_logger()
    worker._stacklevel = 3
    worker._lock = threading.Lock()
    worker._has_initialized = True


class LocalEnvWorker(EnvWorker):
    """Local EnvWorker without Ray/Cluster initialization."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.env.group_name)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_dones_list = []
        self.last_terminations_list = []
        self.last_truncations_list = []
        self.last_intervened_info_list = []

        self.gather_num = 1
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_train = not self.only_eval
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        self.enable_offload = self.cfg.env.get("enable_offload", False)
        self.enable_eval_offload = self.cfg.env.get("enable_offload", False)
        # Compatibility for newer env worker implementations.
        class _LocalPlacement:
            @staticmethod
            def get_world_size(_component: str) -> int:
                return 1

        self._component_placement = _LocalPlacement()
        self.train_dst_ranks = [0]
        self.eval_dst_ranks = [0]
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        else:
            self.train_num_envs_per_stage = 0
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        else:
            self.eval_num_envs_per_stage = 0
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

    def _ensure_compat_attrs(self) -> None:
        """Fill compatibility attrs for env worker variants across commits."""
        if not hasattr(self, "enable_offload"):
            self.enable_offload = self.cfg.env.get("enable_offload", False)
        if not hasattr(self, "enable_eval_offload"):
            self.enable_eval_offload = self.cfg.env.get("enable_offload", False)
        if not hasattr(self, "enable_train"):
            self.enable_train = not getattr(self.cfg.runner, "only_eval", False)
        if not hasattr(self, "enable_eval"):
            self.enable_eval = (
                self.cfg.runner.val_check_interval > 0
                or getattr(self.cfg.runner, "only_eval", False)
            )
        if not hasattr(self, "train_num_envs_per_stage"):
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
                if not getattr(self, "only_eval", False)
                else 0
            )
        if not hasattr(self, "eval_num_envs_per_stage"):
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
                if getattr(self, "enable_eval", False)
                else 0
            )
        if not hasattr(self, "n_train_chunk_steps"):
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        if not hasattr(self, "n_eval_chunk_steps"):
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        if not hasattr(self, "train_dst_ranks"):
            self.train_dst_ranks = [0]
        if not hasattr(self, "eval_dst_ranks"):
            self.eval_dst_ranks = [0]

    def init_worker(self):
        self._ensure_compat_attrs()
        train_env_cls = get_env_cls(
            self.cfg.env.train.env_type, self.cfg.env.train, self.enable_offload
        )
        eval_env_cls = get_env_cls(
            self.cfg.env.eval.env_type, self.cfg.env.eval, self.enable_eval_offload
        )

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.env_list.append(
                    EnvManager(
                        self.cfg.env.train,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=train_env_cls,
                        worker_info=None,
                    )
                )
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                self.eval_env_list.append(
                    EnvManager(
                        self.cfg.env.eval,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        worker_info=None,
                    )
                )

        if not self.only_eval:
            self._init_env()


class LocalRolloutWorker(MultiStepRolloutWorker):
    """Local rollout worker using in-process channels."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.rollout.group_name)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_train = not self.only_eval
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        self.actor_weight_src_rank = 0

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.model_weights_id = ""
        self.count_update = 0

        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        # Compatibility shim:
        # Upstream versions may access self.placement in init_worker/_setup_dst_ranks.
        # For local single-process debug, all component world sizes are 1.
        class _LocalPlacement:
            @staticmethod
            def get_world_size(_component: str) -> int:
                return 1

        self.placement = _LocalPlacement()
        # Compatibility for newer rollout worker implementations.
        self.train_dst_ranks = [0]
        self.eval_dst_ranks = [0]

    def _ensure_compat_attrs(self) -> None:
        """Fill compatibility attrs for rollout worker variants across commits."""
        if not hasattr(self, "enable_train"):
            self.enable_train = not getattr(self.cfg.runner, "only_eval", False)
        if not hasattr(self, "enable_eval"):
            self.enable_eval = (
                self.cfg.runner.val_check_interval > 0
                or getattr(self.cfg.runner, "only_eval", False)
            )
        if not hasattr(self, "n_train_chunk_steps"):
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        if not hasattr(self, "n_eval_chunk_steps"):
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        if not hasattr(self, "train_dst_ranks"):
            self.train_dst_ranks = [0]
        if not hasattr(self, "eval_dst_ranks"):
            self.eval_dst_ranks = [0]
        if not hasattr(self, "placement"):
            class _LocalPlacement:
                @staticmethod
                def get_world_size(_component: str) -> int:
                    return 1

            self.placement = _LocalPlacement()

    def init_worker(self):
        # Do NOT call parent init_worker here.
        # Upstream variants may initialize Cluster()/Ray in parent init_worker.
        self._ensure_compat_attrs()

        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            try:
                self.hf_model.enable_torch_compile(mode=mode)
            except NotImplementedError:
                self._logger.warning(
                    "rollout.enable_torch_compile=True but current policy does not "
                    "support torch.compile; fallback to eager mode."
                )

        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

        self._ensure_compat_attrs()

    def sync_model_from_actor_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.hf_model.load_state_dict(state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(item=chunk_actions, key=f"{self._rank}_{mode}")

    def get_actor_split_num(self):
        send_num = self._world_size * self.num_pipeline_stages
        recv_num = 1
        return compute_split_num(recv_num, send_num)

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i])


class LocalEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Local embodied actor that keeps original FSDP training logic."""

    def __init__(self, cfg: DictConfig):
        _setup_local_worker_attrs(self, cfg.actor.group_name)

        from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

        FSDPModelManager.__init__(self, cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        self.global_step = 0
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        class _LocalPlacement:
            @staticmethod
            def get_world_size(_component: str) -> int:
                return 1

        self._component_placement = _LocalPlacement()
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self._weight_dst_rank_in_rollout = [0]
        self.ref_model = None
        self._value_head_sync_ready = False
        self._shared_ref_param_names: set[str] = set()
        self._enable_mem_log = bool(getattr(self.cfg.actor, "enable_mem_log", False))
        self._update_ready = False
        self._student_param_snapshot = None
        self._student_param_snapshot_init = None
        self._watch_param_names = [
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
        ]
        self._logged_terminal_binary_loss = False

    def _setup_rollout_weight_dst_ranks(self) -> None:
        self._weight_dst_rank_in_rollout = [0]

    def sync_model_to_rollout_state(self) -> dict[str, torch.Tensor]:
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()
        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.ref_model is not None
        ):
            state_dict = self.ref_model.state_dict()
        else:
            state_dict = self.get_model_state_dict(
                cpu_offload=False, full_state_dict=True
            )
        if bool(getattr(self.cfg.rollout, "sync_weights_to_cpu", True)):
            state_dict = {
                k: v.detach().to(device="cpu", non_blocking=True).contiguous()
                if torch.is_tensor(v)
                else v
                for k, v in state_dict.items()
            }
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()
        return state_dict


class LocalEmbodiedRunner:
    """Embodied runner for local single-process debugging."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: LocalEmbodiedFSDPActor,
        rollout: LocalRolloutWorker,
        env: LocalEnvWorker,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env

        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.env_channel = Channel.create("LocalEnv", local=True)
        self.rollout_channel = Channel.create("LocalRollout", local=True)
        self.actor_channel = Channel.create("LocalActor", local=True)

        self.global_step = 0
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        self.actor.init_worker()
        self.rollout.init_worker()
        self.env.init_worker()

    def update_rollout_weights(self):
        state_dict = self.actor.sync_model_to_rollout_state()
        self.rollout.sync_model_from_actor_state(state_dict)

    @staticmethod
    def _pop_worker_execution_times(worker: Any) -> dict[str, float]:
        timer_metrics = getattr(worker, "_timer_metrics", None)
        if not isinstance(timer_metrics, dict):
            return {}
        execution_times = dict(timer_metrics)
        timer_metrics.clear()
        return execution_times

    def _build_train_epoch_env_outputs(self) -> list[EnvOutput]:
        env_output_list = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.env.stage_num):
                self.env.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env.env_list[stage_id].reset()
                dones = (
                    torch.zeros((self.env.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                task_ids = getattr(self.env.env_list[stage_id], "task_ids", None)
                if task_ids is not None:
                    task_ids = torch.as_tensor(task_ids, dtype=torch.long)
                    task_ids = task_ids.view(-1, 1).repeat(
                        1, self.cfg.actor.model.num_action_chunks
                    )
                success_once = getattr(self.env.env_list[stage_id], "success_once", None)
                if success_once is not None:
                    success_once = torch.as_tensor(success_once)
                env_output_list.append(
                    EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        terminations=dones.clone(),
                        truncations=dones.clone(),
                        task_ids=task_ids,
                        success_once=success_once,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                        intervene_actions=None,
                        intervene_flags=None,
                    )
                )
            return env_output_list

        self.env.num_done_envs = 0
        self.env.num_succ_envs = 0
        for stage_id in range(self.env.stage_num):
            success_once = getattr(self.env.env_list[stage_id], "success_once", None)
            env_output_list.append(
                EnvOutput(
                    obs=self.env.last_obs_list[stage_id],
                    rewards=None,
                    dones=self.env.last_dones_list[stage_id],
                    terminations=self.env.last_terminations_list[stage_id],
                    truncations=self.env.last_truncations_list[stage_id],
                    success_once=(
                        torch.as_tensor(success_once) if success_once is not None else None
                    ),
                    intervene_actions=self.env.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.env.last_intervened_info_list[stage_id][1],
                )
            )
        return env_output_list

    def _record_env_metrics(
        self,
        env_metrics: dict[str, list[torch.Tensor]],
        env_info: dict[str, torch.Tensor],
        epoch: int,
    ) -> None:
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    @staticmethod
    def _finalize_metrics(
        metrics: dict[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        finalized = {}
        for key, values in metrics.items():
            if len(values) == 0:
                continue
            finalized[key] = torch.cat(values, dim=0).contiguous().cpu()
        return finalized

    def _run_interact_and_generate(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.rollout.enable_offload:
            self.rollout.reload_model()

        self.rollout.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.rollout.num_pipeline_stages)
        ]
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        env_metrics = defaultdict(list)

        with self.env.worker_timer("interact_step_by_step"):
            for env in self.env.env_list:
                env.start_env()

            for epoch in range(self.cfg.algorithm.rollout_epoch):
                env_output_list = self._build_train_epoch_env_outputs()
                last_extracted_obs = [None for _ in range(self.rollout.num_pipeline_stages)]
                last_forward_inputs = [
                    None for _ in range(self.rollout.num_pipeline_stages)
                ]

                for _ in range(n_chunk_steps):
                    for stage_id in range(self.rollout.num_pipeline_stages):
                        env_output = env_output_list[stage_id].to_dict()
                        if last_forward_inputs[stage_id] is not None:
                            last_forward_inputs[stage_id] = (
                                self.rollout.update_intervene_actions(
                                    env_output, last_forward_inputs[stage_id]
                                )
                            )

                        extracted_obs = self.rollout.hf_model.preprocess_env_obs(
                            env_output["obs"]
                        )
                        dones, rewards, real_extracted_obs = (
                            self.rollout.get_dones_and_rewards(env_output, extracted_obs)
                        )

                        with self.rollout.worker_timer("generate_step_by_step"):
                            actions, result = self.rollout.predict(
                                extracted_obs, task_ids=env_output.get("task_ids")
                            )

                        self.rollout.buffer_list[stage_id].append_result(
                            ChunkStepResult(
                                prev_logprobs=result["prev_logprobs"],
                                prev_values=result["prev_values"],
                                dones=dones,
                                truncations=env_output["truncations"],
                                terminations=env_output["terminations"],
                                rewards=rewards,
                                success_once=env_output.get("success_once"),
                                forward_inputs=last_forward_inputs[stage_id],
                            )
                        )
                        if last_extracted_obs[stage_id] is not None and hasattr(
                            self.rollout.hf_model, "q_head"
                        ):
                            self.rollout.buffer_list[stage_id].add_transition(
                                last_extracted_obs[stage_id], real_extracted_obs
                            )
                        last_extracted_obs[stage_id] = extracted_obs
                        last_forward_inputs[stage_id] = result["forward_inputs"]

                        next_env_output, env_info = self.env.env_interact_step(
                            actions, stage_id
                        )
                        env_output_list[stage_id] = next_env_output
                        self._record_env_metrics(env_metrics, env_info, epoch)

                for stage_id in range(self.rollout.num_pipeline_stages):
                    env_output = env_output_list[stage_id].to_dict()
                    last_forward_inputs[stage_id] = self.rollout.update_intervene_actions(
                        env_output, last_forward_inputs[stage_id]
                    )
                    extracted_obs = self.rollout.hf_model.preprocess_env_obs(
                        env_output["obs"]
                    )
                    dones, rewards, real_extracted_obs = (
                        self.rollout.get_dones_and_rewards(env_output, extracted_obs)
                    )

                    self.rollout.buffer_list[stage_id].dones.append(dones)
                    self.rollout.buffer_list[stage_id].truncations.append(
                        env_output["truncations"]
                    )
                    self.rollout.buffer_list[stage_id].terminations.append(
                        env_output["terminations"]
                    )
                    self.rollout.buffer_list[stage_id].rewards.append(rewards)
                    if env_output.get("success_once") is not None:
                        self.rollout.buffer_list[stage_id].success_once.append(
                            env_output["success_once"].cpu().contiguous()
                        )
                    self.rollout.buffer_list[stage_id].forward_inputs.append(
                        put_tensor_device(last_forward_inputs[stage_id], "cpu")
                    )

                    with self.rollout.worker_timer("bootstrap_step_by_step"):
                        _, result = self.rollout.predict(
                            extracted_obs, task_ids=env_output.get("task_ids")
                        )

                    if "prev_values" in result:
                        self.rollout.buffer_list[stage_id].prev_values.append(
                            result["prev_values"].cpu().contiguous()
                        )
                    if hasattr(self.rollout.hf_model, "q_head"):
                        self.rollout.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )

                self.env.last_obs_list = [env_output.obs for env_output in env_output_list]
                self.env.last_dones_list = [env_output.dones for env_output in env_output_list]
                self.env.last_truncations_list = [
                    env_output.truncations for env_output in env_output_list
                ]
                self.env.last_terminations_list = [
                    env_output.terminations for env_output in env_output_list
                ]
                self.env.last_intervened_info_list = [
                    (env_output.intervene_actions, env_output.intervene_flags)
                    for env_output in env_output_list
                ]
                self.env.finish_rollout()

            for env in self.env.env_list:
                if self.env.enable_offload and hasattr(env, "close"):
                    env.close()
                env.stop_env()

        env_metrics_raw = self._finalize_metrics(env_metrics)
        for stage_id in range(self.rollout.num_pipeline_stages):
            self.rollout.send_rollout_batch(self.actor_channel, stage_id)
        self.actor.recv_rollout_batch(input_channel=self.actor_channel)
        if self.rollout.enable_offload:
            self.rollout.offload_model()
        rollout_metrics = self.actor.compute_advantages_and_returns()
        return env_metrics_raw, rollout_metrics

    def evaluate(self):
        if self.rollout.enable_offload:
            self.rollout.reload_model()

        eval_metrics_raw = defaultdict(list)
        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        with self.env.worker_timer("evaluate_step_by_step"):
            for stage_id in range(self.env.stage_num):
                self.env.eval_env_list[stage_id].start_env()

            for _ in range(self.cfg.algorithm.eval_rollout_epoch):
                env_output_list = []
                for stage_id in range(self.env.stage_num):
                    self.env.eval_env_list[stage_id].is_start = True
                    extracted_obs, infos = self.env.eval_env_list[stage_id].reset()
                    success_once = getattr(
                        self.env.eval_env_list[stage_id], "success_once", None
                    )
                    env_output_list.append(
                        EnvOutput(
                            obs=extracted_obs,
                            final_obs=infos["final_observation"]
                            if "final_observation" in infos
                            else None,
                            success_once=(
                                torch.as_tensor(success_once)
                                if success_once is not None
                                else None
                            ),
                        )
                    )

                for eval_step in range(n_chunk_steps):
                    for stage_id in range(self.env.stage_num):
                        env_output = env_output_list[stage_id].to_dict()
                        extracted_obs = self.rollout.hf_model.preprocess_env_obs(
                            env_output["obs"]
                        )
                        with self.rollout.worker_timer("evaluate_generate_step_by_step"):
                            actions, _ = self.rollout.predict(
                                extracted_obs,
                                mode="eval",
                                task_ids=env_output.get("task_ids"),
                            )
                        next_env_output, env_info = self.env.env_evaluate_step(
                            actions, stage_id
                        )
                        for key, value in env_info.items():
                            eval_metrics_raw[key].append(value)
                        if eval_step != n_chunk_steps - 1:
                            env_output_list[stage_id] = next_env_output

                self.env.finish_rollout(mode="eval")

            for stage_id in range(self.env.stage_num):
                if self.env.enable_offload and hasattr(self.env.eval_env_list[stage_id], "close"):
                    self.env.eval_env_list[stage_id].close()
                self.env.eval_env_list[stage_id].stop_env()

        if self.rollout.enable_offload:
            self.rollout.offload_model()

        eval_metrics = compute_evaluate_metrics([self._finalize_metrics(eval_metrics_raw)])
        return eval_metrics

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step)

    def run(self):
        start_time = time.time()
        for step in range(self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                with self.timer("generate_rollouts"):
                    env_metrics_raw, rollout_metrics_raw = (
                        self._run_interact_and_generate()
                    )

                training_metrics_raw = self.actor.run_training()
                self.global_step += 1

                run_val, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            time_metrics.update(
                {
                    f"time/env/{k}": v
                    for k, v in self._pop_worker_execution_times(self.env).items()
                }
            )
            time_metrics.update(
                {
                    f"time/rollout/{k}": v
                    for k, v in self._pop_worker_execution_times(self.rollout).items()
                }
            )
            time_metrics.update(
                {
                    f"time/actor/{k}": v
                    for k, v in self._pop_worker_execution_times(self.actor).items()
                }
            )

            env_metrics = compute_evaluate_metrics([env_metrics_raw])
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            rollout_metrics = {f"rollout/{k}": v for k, v in rollout_metrics_raw.items()}
            training_metrics = {
                f"train/{k}": v for k, v in training_metrics_raw.items()
            }

            self.metric_logger.log(env_metrics, step)
            self.metric_logger.log(rollout_metrics, step)
            self.metric_logger.log(time_metrics, step)
            self.metric_logger.log(training_metrics, step)

            logging_metrics = {}
            logging_metrics.update(time_metrics)
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            print_metrics_table(step, self.max_steps, start_time, logging_metrics, 0)

        self.metric_logger.finish()


def _validate_local_cfg(cfg: DictConfig) -> DictConfig:
    if cfg.runner.task_type != "embodied":
        raise ValueError("Local debug entry only supports runner.task_type=embodied.")
    if cfg.actor.training_backend != "fsdp":
        raise ValueError("Local debug entry only supports actor.training_backend=fsdp.")
    if cfg.cluster.num_nodes != 1:
        raise ValueError("Local debug entry requires cluster.num_nodes == 1.")

    with open_dict(cfg):
        cfg.runner.weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        cfg.actor = validate_fsdp_cfg(cfg.actor)
        if cfg.env.train.env_type == "libero":
            cfg.env.train.debug_use_dummy_vector_env = True
        if cfg.env.eval.env_type == "libero":
            cfg.env.eval.debug_use_dummy_vector_env = True

    assert cfg.runner.weight_sync_interval > 0, (
        "runner.weight_sync_interval must be greater than 0."
    )
    assert cfg.env.train.total_num_envs > 0, "env.train.total_num_envs must be > 0."
    assert cfg.env.eval.total_num_envs > 0, "env.eval.total_num_envs must be > 0."
    assert cfg.env.train.total_num_envs % cfg.rollout.pipeline_stage_num == 0, (
        "env.train.total_num_envs must be divisible by rollout.pipeline_stage_num."
    )
    assert cfg.env.eval.total_num_envs % cfg.rollout.pipeline_stage_num == 0, (
        "env.eval.total_num_envs must be divisible by rollout.pipeline_stage_num."
    )
    return cfg


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = _validate_local_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    with _init_single_process_dist():
        actor = LocalEmbodiedFSDPActor(cfg)
        rollout = LocalRolloutWorker(cfg)
        env = LocalEnvWorker(cfg)
        runner = LocalEmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
        runner.init_workers()
        runner.run()


if __name__ == "__main__":
    main()
