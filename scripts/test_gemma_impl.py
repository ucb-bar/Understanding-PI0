import torch
from transformers.models.gemma.modeling_gemma import (
    GemmaRMSNorm,
    GemmaMLP,
    GemmaRotaryEmbedding,
    GemmaAttention,
)

from understanding_pi0.gemma_blocks import (
    gemma_rms_norm_forward,
    gemma_mlp_forward,
    gemma_rotary_embedding_forward,
    gemma_attention_forward,
)


class Pi0GemmaConfig:
    max_position_embeddings = 1024
    hidden_size = 1024
    """ Dimension of the hidden representations. """
    intermediate_size = 2048
    """ Dimension of the MLP representations. """
    num_hidden_layers = 18
    num_attention_heads = 8
    head_dim = 256
    num_key_value_heads = 1
    hidden_act = "gelu"

    rms_norm_eps = 1e-6
    rope_theta = 10000.0

    adarms_cond_dim = None

    attention_dropout = 0.0
    attention_bias = False
    _attn_implementation = "eager"


cfg = Pi0GemmaConfig()

SEQ_LEN = 64


def check_result(golden_output, x_output, atol=1e-3, extra_message=""):
    print(
        "  ",
        "✅" if torch.allclose(x_output, golden_output, atol=atol) else "❌",
        "- max error =",
        torch.max(torch.abs(x_output - golden_output)),
        extra_message,
    )


def test_gemma_rms_norm():
    print("Testing gemma_rms_norm_forward...")

    golden_rms_norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    x = torch.randn(SEQ_LEN, cfg.hidden_size)
    golden_rms_norm_output, _ = golden_rms_norm(x)
    x_norm = gemma_rms_norm_forward(x, cfg.rms_norm_eps, cfg.adarms_cond_dim)
    check_result(golden_rms_norm_output, x_norm)


def test_gemma_mlp():
    print("Testing gemma_mlp_forward...")

    golden_mlp = GemmaMLP(cfg)

    x = torch.randn(SEQ_LEN, cfg.hidden_size)
    golden_mlp_output = golden_mlp(x)
    x_down = gemma_mlp_forward(x, golden_mlp.gate_proj.weight, golden_mlp.up_proj.weight, golden_mlp.down_proj.weight)

    check_result(golden_mlp_output, x_down)


def test_gemma_rotary_embedding():
    print("Testing gemma_rotary_embedding_forward...")

    golden_rotary_embedding = GemmaRotaryEmbedding(cfg)
    x = torch.randn(SEQ_LEN, cfg.hidden_size)
    position_ids = torch.arange(0, SEQ_LEN).unsqueeze(0)
    golden_rotary_embedding_output = golden_rotary_embedding(x, position_ids)
    x_rotary_embedding = gemma_rotary_embedding_forward(x, position_ids, cfg.head_dim)

    check_result(golden_rotary_embedding_output, x_rotary_embedding)


def test_gemma_attention():
    print("Testing gemma_attention_forward...")

    golden_attention = GemmaAttention(cfg, 0)
    golden_rotary_embedding = GemmaRotaryEmbedding(cfg)
    seq_len = 8
    x = torch.randn(1, seq_len, cfg.hidden_size)
    position_ids = torch.arange(0, seq_len).unsqueeze(0)
    # Create position embeddings using the rotary embedding module
    position_embeddings = golden_rotary_embedding(x, position_ids)
    golden_attention_output, _ = golden_attention(x, position_embeddings, attention_mask=None)
    x_attention, _ = gemma_attention_forward(
        x,
        position_embeddings,
        golden_attention.q_proj.weight,
        golden_attention.k_proj.weight,
        golden_attention.v_proj.weight,
        golden_attention.o_proj.weight,
        None,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        False,
    )

    check_result(golden_attention_output, x_attention)


if __name__ == "__main__":
    torch.manual_seed(42)

    # test_gemma_rms_norm()
    test_gemma_mlp()
    # test_gemma_rotary_embedding()
    # test_gemma_attention()
