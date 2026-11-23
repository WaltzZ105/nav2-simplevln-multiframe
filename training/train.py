"""Run NAV2 dual-head imitation training with the ms-swift SFT Trainer."""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel, GenerationConfig, Trainer, TrainerCallback
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from bridge_model import Model
from nav2_data import NavigationCollator, NavigationDataset, swift_template
import nav2_factorized_20261030 as nav2_loss
import nav2_lora_model_20261028 as lora


class NavigationModel(Model):
    @property
    def device(self):
        return self.input_projection.weight.device


def load_nav2_weights(model, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    lora.load_adapter(model, saved["adapter"])
    model.input_projection.load_state_dict(saved["input_projection"], strict=True)
    model.head.load_state_dict(saved["head"], strict=True)
    return saved


class NavigationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, navigation_collator, protocol_hash, recompute, skip_recompute_layers, **kwargs):
        self.navigation_collator = navigation_collator
        self.protocol_hash = protocol_hash
        self.recompute = recompute
        self.skip_recompute_layers = skip_recompute_layers
        super().__init__(*args, **kwargs)
        # The loss is normalized over all four ranks and all microbatches in an update.
        self.model_accepts_loss_kwargs = True
        self.next_exposure = None

    def _get_data_collator(self, args, template):
        return self.navigation_collator

    def _prepare_inputs(self, inputs):
        return Trainer._prepare_inputs(self, inputs)

    def _prepare_gradient_checkpointing(self, model):
        # bridge_model already enables non-reentrant recomputation for the text backbone.
        self.args.gradient_checkpointing = False

    def _get_num_items_in_batch(self, batch_samples, device):
        counts = torch.zeros(2, dtype=torch.long, device=device)
        for batch in batch_samples:
            counts[0] += len(batch["exposures"])
            counts[1] += batch["mask"].sum().to(device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(counts)
        return counts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        heatmap, actions = model(inputs["inputs"])
        mask = inputs["mask"]
        hm_loss, action_loss, _ = nav2_loss.loss(heatmap, actions, inputs["heatmap"], inputs["actions"], mask)
        sample_count, action_count = num_items_in_batch
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        exposure = inputs["exposures"].tolist()
        if self.next_exposure is None:
            self.next_exposure = (self.state.global_step * self.args.train_batch_size * world_size
                                  * self.args.gradient_accumulation_steps
                                  + self.args.process_index)
        expected = list(range(self.next_exposure, self.next_exposure + world_size * len(exposure), world_size))
        if exposure != expected:
            raise ValueError(f"Sample order changed: expected {expected}, received {exposure}")
        self.next_exposure = exposure[-1] + world_size
        loss = world_size * (hm_loss * len(exposure) / sample_count + action_loss * mask.sum() / action_count)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite navigation loss")
        if self.state.global_step < 3:
            print(json.dumps({"global_step": self.state.global_step, "rank": self.args.process_index,
                              "exposures": exposure, "frame_counts": inputs["frame_counts"],
                              "tokens": inputs["inputs"]["input_ids"].shape[-1], "nodes": inputs["nodes"],
                              "heatmap_loss": float(hm_loss.detach()), "action_loss": float(action_loss.detach())}), flush=True)
        return (loss, {"heatmap": heatmap, "actions": actions}) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        path = directory / "nav2.pt"
        temporary = path.with_suffix(".partial")
        torch.save(model.checkpoint(self.state.global_step, {
            "protocol_sha256": self.protocol_hash, "training_framework": "simplevln_ms_swift_sft",
            "history_max": 8, "observation_stride": 1, "target_updates": self.args.max_steps,
            "micro_batch": self.args.per_device_train_batch_size, "world_size": self.args.world_size,
            "initialization": "nav2_cuda_lora", "recompute": self.recompute,
            "skip_recompute_layers": self.skip_recompute_layers,
        }), temporary)
        temporary.replace(path)
        torch.save(self.args, directory / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        state = load_nav2_weights(target, Path(resume_from_checkpoint) / "nav2.pt")
        expected = {"protocol_sha256": self.protocol_hash, "target_updates": self.args.max_steps,
                    "world_size": self.args.world_size, "initialization": "nav2_cuda_lora"}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume checkpoint and training configuration differ")
        if state.get("observation_stride") != 1:
            raise ValueError("Resume requires an atomic-observation checkpoint")
        old_micro = state["micro_batch"]
        old_recompute = state.get("recompute", True)
        if (old_micro, old_recompute, state.get("skip_recompute_layers", 0)) != (
                self.args.per_device_train_batch_size, self.recompute, self.skip_recompute_layers):
            raise ValueError("Resume requires the original micro-batch and gradient recomputation setting")


class VerifyUpdate(TrainerCallback):
    def __init__(self, model, stop_after, save_at):
        self.model = model
        self.stop_after = stop_after
        self.save_at = set(save_at)

    def on_train_begin(self, args, state, control, **kwargs):
        self.before = {
            "hm_head.weight": self.model.head.hm_head.weight.detach().clone(),
            "ac_head.weight": self.model.head.ac_head.weight.detach().clone(),
        }
        self.adapter = next(value for name, value in self.model.backbone.named_parameters()
                            if name.endswith(".lora_B"))
        self.projection = self.model.input_projection.weight
        self.before["lora_B"] = self.adapter.detach().clone()
        self.before["input_projection"] = self.projection.detach().clone()
        self.first_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if self.before:
            params = dict(self.model.head.named_parameters())
            params.update(lora_B=self.adapter, input_projection=self.projection)
            updates = {name: float((params[name].detach() - value).norm()) for name, value in self.before.items()}
            if all(delta > 0 for delta in updates.values()):
                if state.is_world_process_zero:
                    print(json.dumps({"updated_step": state.global_step, "head_update_norms": updates,
                                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}), flush=True)
                self.before.clear()
            elif state.global_step >= self.first_step + 2 or self.stop_after == state.global_step:
                raise ValueError(f"A navigation head did not update: {updates}")
        if self.stop_after is not None and state.global_step >= self.stop_after:
            control.should_save = True
            control.should_training_stop = True
        if state.global_step in self.save_at:
            control.should_save = True
        return control


class MeasureUpdates(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.previous = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        metrics = torch.tensor([now - self.previous, torch.cuda.max_memory_allocated() / 2**30,
                                torch.cuda.max_memory_reserved() / 2**30], dtype=torch.float64, device=args.device)
        if dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved = metrics.tolist()
        self.previous = now
        if state.is_world_process_zero:
            print(json.dumps({"perf_update": state.global_step, "wall_seconds": seconds,
                              "max_allocated_gib": allocated, "max_reserved_gib": reserved}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-at", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--skip-recompute-layers", type=int, default=0)
    parser.add_argument("--measure-updates", action="store_true")
    opts = parser.parse_args()

    dataset = NavigationDataset(opts.protocol)
    protocol = dataset.protocol
    template = swift_template(protocol["processor"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    random.seed(protocol["seed"])
    backbone = AutoModel.from_pretrained(protocol["model_path"], dtype=torch.bfloat16,
                                         local_files_only=True).to(device).eval()
    model = NavigationModel(backbone).to(device)
    if opts.disable_recompute:
        backbone.gradient_checkpointing_disable()
    if opts.skip_recompute_layers:
        layers = backbone.language_model.layers
        count = opts.skip_recompute_layers
        if opts.disable_recompute or not 1 <= count < len(layers):
            raise ValueError("Partial recomputation requires a proper subset of text layers")
        selected = [((2 * i + 1) * len(layers)) // (2 * count) for i in range(count)]
        for index in selected:
            if not layers[index].gradient_checkpointing:
                raise ValueError(f"Expected recomputation to be enabled at layer {index}")
            layers[index].gradient_checkpointing = False
        if rank == 0:
            print(json.dumps({"text_layers_without_recomputation": selected}), flush=True)
    model.config = backbone.config
    model.generation_config = GenerationConfig.from_model_config(backbone.config)
    model.model_meta = template.model_meta
    model.model_info = template.model_info
    torch.backends.cuda.matmul.allow_tf32 = True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_global = world_size * opts.micro_batch
    if protocol["global_batch"] % micro_global or protocol["sample_exposures"] % world_size:
        raise ValueError("The global batch and frozen exposure count must divide evenly across the requested ranks")
    steps = opts.steps or protocol["updates"]
    args = Seq2SeqTrainingArguments(
        output_dir=str(opts.output), max_steps=steps, per_device_train_batch_size=opts.micro_batch,
        gradient_accumulation_steps=protocol["global_batch"] // micro_global,
        learning_rate=1e-5, warmup_steps=round(.05 * steps), lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6}, seed=protocol["seed"], data_seed=protocol["seed"],
        train_dataloader_shuffle=False, remove_unused_columns=False, check_model=False,
        dataloader_num_workers=opts.workers, dataloader_persistent_workers=bool(opts.workers),
        save_strategy="steps", save_steps=opts.save_steps, report_to=[], logging_steps=20 if not opts.stop_after else 1,
        gradient_checkpointing=False, bf16=False, fp16=False, ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False, disable_tqdm=True,
    )
    trainer = NavigationTrainer(model=model, args=args, template=template, train_dataset=dataset,
                                navigation_collator=NavigationCollator(template), protocol_hash=dataset.protocol_hash,
                                recompute=not opts.disable_recompute, skip_recompute_layers=opts.skip_recompute_layers)
    trainer.add_callback(VerifyUpdate(model, opts.stop_after, opts.save_at))
    if opts.measure_updates:
        trainer.add_callback(MeasureUpdates())
    trainer.train(resume_from_checkpoint=str(opts.resume) if opts.resume else None)
    if trainer.is_world_process_zero():
        print(json.dumps({"completed_update": trainer.state.global_step, "steps": steps}), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
