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


def run_once(preprocessor, policy, postprocessor):
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


torch.cuda.memory._record_memory_history()

policy = PI0Policy(config)

policy.to(DEVICE)
policy.config.device = DEVICE

preprocessor, postprocessor = make_pi0_pre_post_processors(
    config=policy.config, dataset_stats=DATASET_STATS
)
run_once(preprocessor, policy, postprocessor)
torch.cuda.memory._dump_snapshot("pi0_memory_snapshot.pickle")
