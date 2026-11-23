"""Bind the NAV2 task when explicitly loaded by megatron sft."""

from swift.megatron.pipelines.train.sft import MegatronSft

from nav2_megatron_trainer import Nav2MegatronTrainer


def prepare_nav2_trainer(pipeline):
    return Nav2MegatronTrainer(pipeline.args, pipeline.template)


MegatronSft.prepare_trainer = prepare_nav2_trainer
