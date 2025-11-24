from typing import Optional, Any, Sequence, List, Literal
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path, load_callable
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class DatasetGeneratorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    output_root: str
    initial_max_bits: Optional[int] = None
    bits_increment: int = 1
    regen_interval: Optional[int] = 10000
    samples_per_bit: int = 1000
    min_bits: Optional[int] = None
    test_samples_per_bit: int = 0
    refresh_test_split: bool = True
    seed_offset: int = 0


class LRSchedulerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: Literal["reduce_on_plateau"]
    metric: str
    mode: Literal["min", "max"] = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 0.0
    lr_num_cycles: float = 0.5
    lr_scheduler: Optional[LRSchedulerConfig] = None

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    dataset_generator: Optional[DatasetGeneratorConfig] = None
    stage_advance_metric: Optional[str] = None
    stage_advance_threshold: Optional[float] = None
    stage_advance_comparison: str = "lt"

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: List[float]
    lr_schedulers: Sequence[Any]
    carry: Any

    step: int
    total_steps: int
    stage_step: int
    stage_max_steps: int


@dataclass
class PlateauSchedulerState:
    best: Optional[float] = None
    num_bad_epochs: int = 0
    cooldown_counter: int = 0


class DynamicDatasetManager:
    def __init__(self, config: PretrainConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.dataset_cfg = config.dataset_generator
        self.generator_fn = None
        if self.dataset_cfg is not None:
            self.generator_fn = load_callable(self.dataset_cfg.name)
        self.current_stage: Optional[int] = None

    def enabled(self) -> bool:
        return self.generator_fn is not None and self.dataset_cfg is not None

    def ensure_stage(self, stage_idx: int):
        if not self.enabled():
            return False
        if self.current_stage == stage_idx:
            return False
        stage_info = self._generate_stage(stage_idx)
        self.current_stage = stage_idx
        return stage_info

    def _generate_stage(self, stage_idx: int):
        assert self.dataset_cfg is not None and self.generator_fn is not None
        cfg = self.dataset_cfg

        start_bits = cfg.initial_max_bits
        if start_bits is None:
            if cfg.min_bits is not None:
                start_bits = cfg.min_bits
            else:
                raise ValueError("dataset_generator requires either initial_max_bits or min_bits to be set.")

        max_bits = start_bits + stage_idx * cfg.bits_increment
        train_size = max_bits * max(1, cfg.samples_per_bit)
        test_size = max_bits * cfg.test_samples_per_bit if cfg.test_samples_per_bit > 0 else 0
        output_dir = os.path.join(cfg.output_root, f"stage_{stage_idx:04d}_bits_{max_bits}")
        min_bits = cfg.min_bits if cfg.min_bits is not None else max_bits
        min_bits = min(min_bits, max_bits)
        extra_kwargs = dict(cfg.__pydantic_extra__ or {})
        generator_kwargs = {
            **extra_kwargs,
            "output_dir": output_dir,
            "max_bits": max_bits,
            "min_bits": min_bits,
            "train_size": train_size,
            "test_size": test_size,
            "seed": self.config.seed + cfg.seed_offset + stage_idx,
        }

        if self.rank == 0:
            print(f"[DynamicDataset] Stage {stage_idx}: max_bits={max_bits}, train_examples={train_size}")
            os.makedirs(output_dir, exist_ok=True)
            self.generator_fn(**generator_kwargs)

        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        self.config.data_paths = [output_dir]
        if not self.config.data_paths_test or cfg.refresh_test_split:
            if test_size > 0:
                self.config.data_paths_test = [output_dir]
            elif cfg.refresh_test_split:
                self.config.data_paths_test = []

        return {
            "stage": stage_idx,
            "output_dir": output_dir,
            "max_bits": max_bits,
            "min_bits": min_bits,
            "train_size": train_size,
            "test_size": test_size,
        }


def log_dataset_stage_to_wandb(stage_info: Optional[dict], step: int):
    if stage_info is None or wandb.run is None:
        return

    wandb.log(
        {
            "dataset/stage": stage_info.get("stage"),
            "dataset/max_bits": stage_info.get("max_bits"),
            "dataset/min_bits": stage_info.get("min_bits"),
            "dataset/train_examples": stage_info.get("train_size"),
            "dataset/test_examples": stage_info.get("test_size"),
        },
        step=step,
    )


def _split_metric_path(metric_path: str) -> List[str]:
    if "." in metric_path:
        return [segment for segment in metric_path.split(".") if segment]
    return [metric_path] if metric_path else []


def get_nested_metric(metrics: Any, metric_path: str) -> Optional[float]:
    """
    Retrieve a nested metric value given a '/' or '.' separated path, e.g. 'test/loss' or 'test.loss'.
    Returns None if any key is missing or if the value cannot be cast to float.
    """
    if metrics is None:
        return None

    path_segments = _split_metric_path(metric_path)
    if not path_segments:
        return None

    value: Any = metrics
    for key in path_segments:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamAtan2(
                model.parameters(),
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamAtan2(
                model.parameters(),
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def create_lr_schedulers(config: PretrainConfig, optimizers: Sequence[torch.optim.Optimizer]) -> Sequence[Any]:
    scheduler_cfg = config.lr_scheduler
    if scheduler_cfg is None:
        return []

    if scheduler_cfg.name == "reduce_on_plateau":
        return [PlateauSchedulerState() for _ in optimizers]

    raise ValueError(f"Unsupported lr_scheduler: {scheduler_cfg.name}")


def step_lr_schedulers(
    config: PretrainConfig,
    train_state: TrainState,
    eval_metrics: Optional[Any],
    train_metrics: Optional[Any],
):
    scheduler_cfg = config.lr_scheduler
    if scheduler_cfg is None or not len(train_state.lr_schedulers):
        return

    metric_value = get_nested_metric(eval_metrics, scheduler_cfg.metric)
    if metric_value is None:
        metric_value = get_nested_metric(train_metrics, scheduler_cfg.metric)

    if metric_value is None:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[LR Scheduler] Metric '{scheduler_cfg.metric}' not found in eval/train metrics; skipping scheduler step.")
        return

    def is_better(val: float, best: Optional[float]) -> bool:
        if best is None:
            return True

        threshold = scheduler_cfg.threshold
        if scheduler_cfg.threshold_mode == "rel":
            threshold = abs(best) * scheduler_cfg.threshold

        if scheduler_cfg.mode == "min":
            return val < (best - threshold)
        return val > (best + threshold)

    for idx, state in enumerate(train_state.lr_schedulers):
        if state.cooldown_counter > 0:
            state.cooldown_counter -= 1
            state.num_bad_epochs = 0

        if is_better(metric_value, state.best):
            state.best = metric_value
            state.num_bad_epochs = 0
            continue

        state.num_bad_epochs += 1
        if state.num_bad_epochs <= scheduler_cfg.patience:
            continue

        new_lr = max(scheduler_cfg.min_lr, train_state.optimizer_lrs[idx] * scheduler_cfg.factor)
        if (train_state.optimizer_lrs[idx] - new_lr) <= scheduler_cfg.eps:
            state.num_bad_epochs = 0
            continue

        train_state.optimizer_lrs[idx] = new_lr
        state.num_bad_epochs = 0
        state.cooldown_counter = scheduler_cfg.cooldown
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[LR Scheduler] Reduced max lr for optimizer {idx} to {new_lr:.5e}")


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    adjusted_total = max(num_training_steps, num_warmup_steps + 1)
    progress = float(current_step - num_warmup_steps) / float(max(1, adjusted_total - num_warmup_steps))
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.stage_step,
        base_lr=base_lr,
        num_warmup_steps=max(0, config.lr_warmup_steps),
        num_training_steps=max(train_state.stage_max_steps, config.lr_warmup_steps + 1),
        min_ratio=config.lr_min_ratio,
        num_cycles=config.lr_num_cycles,
    )


def _estimate_stage_max_steps(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, epochs_per_iter: int) -> int:
    steps_per_epoch = math.ceil(
        (train_metadata.total_groups * train_metadata.mean_puzzle_examples) / max(1, config.global_batch_size)
    )
    steps_per_epoch = max(1, steps_per_epoch)
    epochs_per_iter = max(1, epochs_per_iter)
    return max(1, steps_per_epoch * epochs_per_iter)


def reset_stage_scheduler(
    train_state: TrainState, config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, epochs_per_iter: int
):
    train_state.stage_step = 0
    train_state.stage_max_steps = _estimate_stage_max_steps(config, train_metadata, epochs_per_iter)


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)
    lr_schedulers = create_lr_schedulers(config, optimizers)

    return TrainState(
        step=0,
        total_steps=total_steps,
        lr_schedulers=lr_schedulers,
        stage_step=0,
        stage_max_steps=max(1, total_steps),

        model=model,
        optimizers=optimizers,
        optimizer_lrs=list(optimizer_lrs),
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)

def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def build_eval_components(config: PretrainConfig, rank: int, world_size: int):
    eval_loader = None
    eval_metadata = None
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    evaluators: List[Any] = []
    if eval_metadata is not None:
        try:
            evaluators = create_evaluators(config, eval_metadata)
        except Exception:
            print("No evaluator found")
            evaluators = []

    return eval_loader, eval_metadata, evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    train_state.stage_step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    last_lr = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        last_lr = lr_this_step

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            if last_lr is not None:
                reduced_metrics["train/lr"] = last_lr
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    dataset_manager = DynamicDatasetManager(config, rank=RANK, world_size=WORLD_SIZE)
    iters_per_stage = total_iters
    pending_stage_info = None
    metric_driven_stage = False
    if dataset_manager.enabled():
        pending_stage_info = dataset_manager.ensure_stage(0)
        metric_driven_stage = (
            config.stage_advance_metric is not None and config.stage_advance_threshold is not None
        )
        if not metric_driven_stage:
            regen_interval = config.dataset_generator.regen_interval  # type: ignore[union-attr]
            if regen_interval is None:
                raise ValueError(
                    "dataset_generator.regen_interval must be set when metric-driven stage advancement is disabled."
                )
            assert regen_interval % train_epochs_per_iter == 0, "Dataset regen interval must align with eval interval."
            iters_per_stage = max(1, regen_interval // train_epochs_per_iter)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata, evaluators = build_eval_components(config, rank=RANK, world_size=WORLD_SIZE)
    current_stage_idx = dataset_manager.current_stage if dataset_manager.enabled() else 0
    if current_stage_idx is None:
        current_stage_idx = 0
    current_stage_info = pending_stage_info

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    reset_stage_scheduler(train_state, config, train_metadata, train_epochs_per_iter)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
        log_dataset_stage_to_wandb(current_stage_info, train_state.step)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        if dataset_manager.enabled() and not metric_driven_stage:
            stage_idx = _iter_id // iters_per_stage
            if stage_idx != current_stage_idx:
                new_stage_info = dataset_manager.ensure_stage(stage_idx)
                if new_stage_info:
                    current_stage_info = new_stage_info
                    if RANK == 0:
                        log_dataset_stage_to_wandb(current_stage_info, train_state.step)
                train_loader, train_metadata = create_dataloader(
                    config,
                    "train",
                    test_set_mode=False,
                    epochs_per_iter=train_epochs_per_iter,
                    global_batch_size=config.global_batch_size,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
                reset_stage_scheduler(train_state, config, train_metadata, train_epochs_per_iter)
                eval_loader, eval_metadata, evaluators = build_eval_components(
                    config, rank=RANK, world_size=WORLD_SIZE
                )
                current_stage_idx = stage_idx

        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        train_metrics = None
        eval_metrics_result = None
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            train_metrics = metrics

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if (
            eval_loader is not None
            and eval_metadata is not None
            and _iter_id >= config.min_eval_interval
        ):
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            eval_metrics_result = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and eval_metrics_result is not None:
                wandb.log(eval_metrics_result, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            advance_stage = False
            if (
                dataset_manager.enabled()
                and metric_driven_stage
                and RANK == 0
                and eval_metrics_result is not None
                and config.stage_advance_metric is not None
                and config.stage_advance_threshold is not None
            ):
                metric_value = get_nested_metric(eval_metrics_result, config.stage_advance_metric)
                if metric_value is None:
                    metric_value = get_nested_metric(train_metrics, config.stage_advance_metric)
                if metric_value is not None:
                    print(f"Metric value for {config.stage_advance_metric}: {metric_value}")
                    comparison = (config.stage_advance_comparison or "lt").lower()
                    if comparison == "lt":
                        advance_stage = metric_value < config.stage_advance_threshold
                    elif comparison == "gt":
                        advance_stage = metric_value > config.stage_advance_threshold
                    else:
                        raise ValueError(f"Unsupported stage_advance_comparison: {comparison}")
                else:
                    raise ValueError(f"Metric value for {config.stage_advance_metric} is None. Eval metrics: {eval_metrics_result}, train_metrics: {train_metrics}")

            if (
                dataset_manager.enabled()
                and metric_driven_stage
                and WORLD_SIZE > 1
                and dist.is_initialized()
            ):
                sync_buf = [advance_stage]
                dist.broadcast_object_list(sync_buf, src=0)
                advance_stage = sync_buf[0]

            if dataset_manager.enabled() and metric_driven_stage and advance_stage:
                stage_idx = current_stage_idx + 1
                new_stage_info = dataset_manager.ensure_stage(stage_idx)
                if new_stage_info:
                    current_stage_info = new_stage_info
                    if RANK == 0:
                        log_dataset_stage_to_wandb(current_stage_info, train_state.step)
                train_loader, train_metadata = create_dataloader(
                    config,
                    "train",
                    test_set_mode=False,
                    epochs_per_iter=train_epochs_per_iter,
                    global_batch_size=config.global_batch_size,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
                reset_stage_scheduler(train_state, config, train_metadata, train_epochs_per_iter)
                eval_loader, eval_metadata, evaluators = build_eval_components(
                    config, rank=RANK, world_size=WORLD_SIZE
                )
                current_stage_idx = stage_idx

            if config.ema:
                del train_state_eval

        step_lr_schedulers(config, train_state, eval_metrics_result, train_metrics)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
