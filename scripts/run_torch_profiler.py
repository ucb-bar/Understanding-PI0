from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.policies.pi0 import PI0Config, PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature


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


# warm up
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

report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

profiler.export_chrome_trace(str(report_dir / "chrome_trace.json"))
profiler.export_memory_timeline(str(report_dir / "memory_timeline.html"))
profiler.export_memory_timeline(str(report_dir / "memory_timeline.json.gz"))
profiler.export_stacks(str(report_dir / "stacks.json"))


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
with open(report_dir / "cuda_time_total.log", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=100))

print("Ordered by self CUDA time total:")
print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
with open(report_dir / "self_cuda_time_total.log", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=100))

print("Ordered by FLOPs:")
print(profiler.key_averages().table(sort_by="flops", row_limit=10))
with open(report_dir / "flops.log", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="flops", row_limit=100))

print("Ordered by CUDA memory usage:")
print(profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
with open(report_dir / "self_cuda_memory_usage.log", "w") as f:
    f.write(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=100))


report_data = profiler.key_averages(group_by_input_shape=True)
report_data.sort(key=lambda x: x.self_device_time_total, reverse=True)

kernels = []
kernel_names = []
time_usage = []

remaining_time = 0

for item in report_data:
    if len(kernels) < 20:
        if "aten" in item.key:
            kernels.append(item)
            kernel_names.append(f"{item.key} {item.input_shapes}")
            time_usage.append(item.self_device_time_total)  # in us

    else:
        remaining_time += item.self_device_time_total

kernel_names.append("Remaining")
time_usage.append(remaining_time)

time_usage = np.array(time_usage, dtype=np.float32)
time_usage *= 1e-3  # convert to ms


fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(kernel_names, time_usage)
ax.set_xlabel("Kernel")
ax.set_ylabel("Time (ms)")
ax.set_xticklabels(kernel_names, rotation=45, ha="right")
ax.set_title("Kernel percentage by time")
plt.tight_layout()
fig.savefig(report_dir / "time_usage.png", bbox_inches='tight', pad_inches=0.5)

# also do a pie chart of the time usage
fig, ax = plt.subplots(figsize=(12, 8))
ax.pie(time_usage, labels=kernel_names, autopct="%1.1f%%")
ax.set_title("Kernel percentage by time")
plt.tight_layout()
fig.savefig(report_dir / "time_usage_pie.png", bbox_inches='tight', pad_inches=0.5)
