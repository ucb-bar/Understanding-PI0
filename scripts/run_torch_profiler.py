import torch
from lerobot.policies.pi0 import PI0Config, PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json


MAX_ACTION_DIM = 16
MAX_STATE_DIM = 32
ACTION_DIM = 7
STATE_DIM = 13
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 48  # Default for PI0 (non-pi05)
DEVICE = "cuda"  # Use CPU to avoid memory issues for testing
DTYPE = "bfloat16"

DATASET_STATS = {
    "observation.state": {
        "mean": torch.zeros(STATE_DIM),
        "std": torch.ones(STATE_DIM),
        "q01": torch.zeros(STATE_DIM),
        "q99": torch.ones(STATE_DIM),
    },
    "action": {
        "mean": torch.zeros(ACTION_DIM),
        "std": torch.ones(ACTION_DIM),
        "q01": torch.zeros(ACTION_DIM),
        "q99": torch.ones(ACTION_DIM),
    },
    "images": {
        "base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "left_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "right_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    },
}

INPUT_FEATURES = {
    "observation.state": PolicyFeature(
        type=FeatureType.STATE,
        shape=(STATE_DIM,),
    ),
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(ACTION_DIM,),
    ),
    "observation.images.base_0_rgb": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),
    ),
    "observation.images.left_wrist_0_rgb": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),
    ),
    "observation.images.right_wrist_0_rgb": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224),
    ),
}

OUTPUT_FEATURES = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(ACTION_DIM,),
    ),
}

batch_size = 1  # Reduce batch size for testing
device = DEVICE

# Use the exact same prompt for both implementations
prompt = "Pick up the red block and place it in the bin"

config = PI0Config(
    input_features=INPUT_FEATURES,
    output_features=OUTPUT_FEATURES,
    max_action_dim=MAX_ACTION_DIM,
    max_state_dim=MAX_STATE_DIM,
    dtype=DTYPE,
)
policy = PI0Policy(config)

policy.to(DEVICE)
policy.config.device = DEVICE


# preprocessor: DataProcessorPipeline(name="policy_preprocessor", steps=6: [
#   RenameObservationsProcessorStep,
#   AddBatchDimensionProcessorStep,
#   Pi0NewLineProcessor,
#   TokenizerProcessorStep,
#   DeviceProcessorStep,
#   NormalizerProcessorStep,
# ])
# postprocessor: DataProcessorPipeline(name="policy_postprocessor", steps=2: [
#   UnnormalizerProcessorStep,
#   DeviceProcessorStep
# ])
preprocessor, postprocessor = make_pi0_pre_post_processors(
    config=policy.config, dataset_stats=DATASET_STATS
)

print(policy)


def run_once():
    observations = {
        "observation.state": torch.randn(
            batch_size, STATE_DIM, dtype=torch.float32, device=device
        ),
        # "action": torch.randn(
        #     batch_size, ACTION_HORIZON, ACTION_DIM, dtype=torch.float32, device=device
        # ),
        # Create images in [0, 1] range as expected by LeRobot (will be converted to [-1, 1] internally)
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "observation.images.left_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "observation.images.right_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        # Add the task prompt for LeRobot - provide as list with single element to trigger expansion
        "task": [prompt for _ in range(batch_size)],
    }
    # observations_processed:
    #   action: <(1, 50, 7), float32>
    #   observation.state: <(1, 13), float32>
    #   observation.images.XX: <(1, 3, 224, 224), float32>
    #   observation.language.tokens: <(1, 48), int64>
    #   observation.language.attention_mask: <(1, 48), bool>
    observations_processed = preprocessor(observations)

    # actions: <(1, 50, 7), float32>
    actions = policy.predict_action_chunk(observations_processed)

    actions_processed = postprocessor(actions)

    return actions_processed


for _ in range(10):
    run_once()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as profiler:
    for _ in range(1):
        run_once()

profiler.export_chrome_trace("pi0_trace.json")
profiler.export_memory_timeline("pi0_memory_timeline.html")
profiler.export_stacks("pi0_stacks.json")


model_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
vision_tower_params = sum(p.numel() for p in policy.model.paligemma_with_expert.paligemma.model.vision_tower.parameters())
language_model_params = sum(p.numel() for p in policy.model.paligemma_with_expert.paligemma.model.language_model.parameters())
action_model_params = sum(p.numel() for p in policy.model.paligemma_with_expert.gemma_expert.parameters())

total_flops = profiler.key_averages().total_average().flops

print(f"Model parameters: {model_params} ({model_params / (1e9):.2f} B)")
print(f"Model trainable parameters: {trainable_params} ({trainable_params / (1e9):.2f} B)")
print(f" - vision tower: vision_tower_params: {vision_tower_params} ({vision_tower_params / (1e6):.2f} M)")
print(f" - language model: language_model_params: {language_model_params} ({language_model_params / (1e6):.2f} M)")
print(f" - action model: action_model_params: {action_model_params} ({action_model_params / (1e6):.2f} M)")
print(f"Model total FLOPs: {total_flops} ({total_flops / (1e12):.2f} T)")

print("Ordered by CUDA time total:")
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
with open("pi0_cuda_time_total.txt", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=100))

print("Ordered by FLOPs:")
print(profiler.key_averages().table(sort_by="flops", row_limit=10))
with open("pi0_flops.txt", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="flops", row_limit=100))

print("Ordered by CUDA memory usage:")
print(profiler.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
with open("pi0_cuda_memory_usage.txt", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=100))
