"""Integrate the NAV2 dual loss into the upstream Megatron SFT loop."""

import itertools
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from swift.megatron.trainers import MegatronTrainer

from adapter import checkpoint, install
from nav2_data import NavigationDataset
from nav2_factorized_20261030 import loss as navigation_loss


class TargetCollator:
    def __init__(self, pack, protocol):
        self.pack = pack
        self.labels = NavigationDataset(protocol)
        self.cache_key = f"nav2:{self.labels.protocol['manifest_sha256']}:"

    def __call__(self, rows):
        batch = self.pack(rows)
        keys = batch.pop("channel", None)
        if not keys or len(keys) != len(batch["seq_lens"]):
            raise ValueError("Official packing dropped a NAV2 target lookup key")
        self.labels._process_handles()
        targets = []
        for key in keys:
            if not isinstance(key, str) or not key.startswith(self.cache_key):
                raise ValueError("Unexpected NAV2 target reference in the cached dataset")
            self.labels._manifest.seek(int(key[len(self.cache_key):]))
            sample = json.loads(self.labels._manifest.readline())
            targets.append(self.labels._labels(sample["nav2"]))
        batch["nav2_targets"] = {name: torch.stack([target[name] for target in targets])
                                 for name in targets[0]}
        if "labels" not in batch:
            batch["labels"] = torch.full_like(batch["input_ids"], -100)
        return batch


class Nav2MegatronTrainer(MegatronTrainer):
    def __init__(self, args, template):
        if (not args.packing or not args.padding_free or args.pipeline_model_parallel_size != 1
                or args.context_parallel_size != 1 or args.calculate_per_token_loss
                or args.task_type != "causal_lm" or args.tuner_type != "full"):
            raise ValueError("NAV2 requires the packed Qwen3.5 Megatron SFT recipe with PP=CP=1")
        if (not args.no_save_optim or not args.no_save_rng or not args.save_safetensors
                or args.mcore_model or args.mcore_adapter):
            raise ValueError("Use the released Stage-1 weights-only checkpoint configuration")
        self.verify_update = os.getenv("NAV2_VERIFY_UPDATE") == "1"
        self.before = None
        super().__init__(args, template)

    def _prepare_peft_model(self, models):
        self.bridge.load_weights(models, self.args.model_dir)
        if len(models) != 1:
            raise ValueError("The NAV2 adapter expects one model chunk per rank")
        model = models[0]
        model.requires_grad_(False)
        self.task = install(model)
        return models

    def _get_data_collator(self):
        return TargetCollator(super()._get_data_collator(), Path(os.environ["NAV2_PROTOCOL"]))

    def train_step(self, train_data_iterator):
        batches = list(itertools.islice(train_data_iterator, self.args.num_microbatches))
        if len(batches) != self.args.num_microbatches:
            raise ValueError("Incomplete Megatron optimizer batch")
        counts = [sum(batch["nav2_targets"]["heatmap"].size(0) for batch in batches),
                  sum(int(batch["nav2_targets"]["mask"].sum()) for batch in batches)]
        self.denominators = torch.tensor(counts, dtype=torch.float32, device="cuda")
        dist.all_reduce(self.denominators, group=mpu.get_data_parallel_group())
        if self.denominators.min().item() < 1:
            raise ValueError("A NAV2 optimizer batch has no supervised actions")
        if self.verify_update and self.before is None:
            self.before = {name: weight.detach().clone() for name, weight in self.tracked().items()}
        result = super().train_step(iter(batches))
        if self.before is not None:
            updates = {name: float((weight.detach() - self.before[name]).norm())
                       for name, weight in self.tracked().items()}
            if all(delta > 0 for delta in updates.values()):
                if dist.get_rank() == 0:
                    print(json.dumps({"head_and_adapter_updates": updates}), flush=True)
                self.verify_update = False
                self.before = None
            elif self.state.iteration + 1 == self.args.train_iters:
                raise ValueError(f"NAV2 parameters did not update during smoke run: {updates}")
        return result

    def tracked(self):
        return {
            "heatmap_head": self.task.head.hm_head.weight,
            "action_head": self.task.head.ac_head.weight,
            "input_projection": self.task.input_projection.weight,
            "lora_B": self.task.adapters[0].lora_B,
        }

    def forward_step(self, data_iterator, model):
        batch = self.get_batch(data_iterator)
        targets = batch.pop("nav2_targets")
        batch.pop("loss_scale", None)
        batch.pop("labels")
        return model(**batch), partial(self.nav2_loss, targets=targets)

    def nav2_loss(self, output, *, targets):
        heatmap = output[..., 0].float()
        actions = output[..., 1:].reshape(*heatmap.shape, 3, 4).float()
        truth = targets["heatmap"]
        valid = targets["mask"]
        heatmap_loss, action_loss, _ = navigation_loss(heatmap, actions, truth, targets["actions"], valid)
        numerators = torch.stack((heatmap_loss * truth.size(0), action_loss * valid.sum()))
        contributions = numerators / self.denominators
        scale = self.args.num_microbatches * mpu.get_data_parallel_world_size()
        total = scale * contributions.sum()
        metrics = scale * contributions.detach()
        return total, {
            "loss": torch.stack([total.detach(), torch.ones_like(total)]),
            "heatmap_loss": torch.stack([metrics[0], torch.ones_like(total)]),
            "action_loss": torch.stack([metrics[1], torch.ones_like(total)]),
        }

    def _log_callback(self, logs, n_steps):
        for component in ("heatmap_loss", "action_loss"):
            if component in logs:
                dist.all_reduce(logs[component], group=mpu.get_data_parallel_group())
        return super()._log_callback(logs, n_steps)

    def save_checkpoint(self):
        super().save_checkpoint()
        nav2 = checkpoint(self.task, self.state.iteration)
        if nav2 is not None:
            output = Path(self.state.last_model_checkpoint) / "nav2.pt"
            temporary = output.with_suffix(".partial")
            torch.save(nav2, temporary)
            temporary.replace(output)
