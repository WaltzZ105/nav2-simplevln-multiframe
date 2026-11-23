"""Use the rendered SimpleVLN observations with unchanged NAV2 archive labels."""

import hashlib
import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from swift.template.templates.qwen import Qwen3_5Template
from torch.utils.data import Dataset


class NavigationTemplate(Qwen3_5Template):
    def _get_position_ids(self, inputs):
        return {}


def swift_template(model_path):
    from swift.model import get_model_processor
    from swift.template.register import TEMPLATE_MAPPING

    _, processor = get_model_processor(str(model_path), load_model=False, use_hf=True, model_type="qwen3_5")
    template = NavigationTemplate(
        processor, TEMPLATE_MAPPING["qwen3_5"], max_length=40960, max_pixels=1003520,
        padding_free=False, enable_thinking=False,
    )
    template.set_mode("transformers")
    return template


class NavigationDataset(Dataset):
    def __init__(self, protocol_path):
        raw = protocol_path.read_bytes()
        self.protocol = json.loads(raw)
        self.protocol_hash = hashlib.sha256(raw).hexdigest()
        self.path = Path(self.protocol["manifest"])
        self.offsets = np.load(self.protocol["offsets"], mmap_mode="r")
        self.sequence = np.load(self.protocol["sequence"], mmap_mode="r")
        self.observations = Path(self.protocol["observation_root"])
        if (len(self.offsets) != self.protocol["unique_rows"]
                or len(self.sequence) != self.protocol["sample_exposures"]):
            raise ValueError("Observation index disagrees with the protocol")
        self._manifest = None
        self._archives = OrderedDict()
        self._pid = None

    def __len__(self):
        return len(self.sequence)

    def _process_handles(self):
        if self._pid != os.getpid():
            for handle in [self._manifest, *self._archives.values()]:
                if handle is not None:
                    handle.close()
            self._manifest = self.path.open("rb")
            self._archives.clear()
            self._pid = os.getpid()

    def _labels(self, target):
        archive = target["archive"]
        handle = self._archives.pop(archive, None)
        if handle is None:
            handle = open(archive, "rb")
        self._archives[archive] = handle
        if len(self._archives) > 32:
            self._archives.popitem(last=False)[1].close()
        ref = target["labels_ref"]
        handle.seek(ref["offset"])
        raw = handle.read(ref["size"])
        if len(raw) != ref["size"]:
            raise EOFError(f"Truncated NAV2 labels for {target['node_id']}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as labels:
            return {
                name: torch.from_numpy(np.asarray(labels[key], dtype=np.float32).copy())
                for name, key in (
                    ("heatmap", "heatmap_target"), ("actions", "action_class_distribution"),
                    ("mask", "action_loss_mask"),
                )
            }

    def __getitem__(self, exposure):
        self._process_handles()
        index = int(self.sequence[exposure])
        self._manifest.seek(int(self.offsets[index]))
        row = json.loads(self._manifest.readline())
        images = []
        for path in row["images"]:
            with Image.open(self.observations / path) as source:
                images.append(source.convert("RGB"))
        example = {
            "messages": [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["messages"][0]["value"]},
            ],
            "images": images,
        }
        target = row["nav2"]
        return example, self._labels(target), exposure, target["node_id"]


class NavigationCollator:
    def __init__(self, template):
        self.template = template

    def __call__(self, batch):
        examples, labels, exposures, nodes = zip(*batch)
        encoded = [self.template.encode(example) for example in examples]
        longest = max(len(item["input_ids"]) for item in encoded)
        padding_to = ((longest + 255) // 256) * 256
        inputs = self.template.data_collator(encoded, padding_to=padding_to)
        model_inputs = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
        return {
            "inputs": {key: value for key, value in inputs.items() if key in model_inputs},
            **{key: torch.stack([row[key] for row in labels]) for key in labels[0]},
            "exposures": torch.tensor(exposures), "nodes": nodes,
            "frame_counts": [len(example["images"]) for example in examples],
        }
