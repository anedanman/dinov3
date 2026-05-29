# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Map a step-based schedule config onto the epoch-based training machinery.

The DINOv3 schedulers express durations as ``epochs * OFFICIAL_EPOCH_LENGTH``.
Setting ``OFFICIAL_EPOCH_LENGTH = 1`` makes every ``*_epochs`` value count in
optimizer steps, so a step-native config (total_steps, warmup_steps, ...) maps
cleanly without touching the scheduler code.
"""

import logging

logger = logging.getLogger("dinov3")


def normalize_step_schedule(cfg):
    """If ``cfg.schedule.enabled``, rewrite the epoch-based keys in terms of steps.

    Mutates and returns ``cfg``.
    """
    sched = cfg.get("schedule", None)
    if sched is None or not sched.get("enabled", False):
        return cfg

    if "schedules" in cfg:
        raise ValueError("schedule.enabled is incompatible with the v2 'schedules' block; use one or the other.")

    total = int(sched.total_steps)
    warmup = int(sched.warmup_steps)
    freeze = int(sched.freeze_last_layer_steps)
    temp_warmup = int(sched.teacher_temp_warmup_steps)

    cfg.train.OFFICIAL_EPOCH_LENGTH = 1
    cfg.optim.epochs = total
    cfg.optim.warmup_epochs = warmup
    cfg.optim.freeze_last_layer_epochs = freeze
    cfg.teacher.warmup_teacher_temp_epochs = temp_warmup

    logger.info(
        "Step-based schedule: total_steps=%d warmup_steps=%d freeze_last_layer_steps=%d teacher_temp_warmup_steps=%d "
        "(OFFICIAL_EPOCH_LENGTH set to 1)",
        total,
        warmup,
        freeze,
        temp_warmup,
    )
    return cfg
