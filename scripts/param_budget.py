#!/usr/bin/env python3
"""CAT-YOKO parameter-budget calculator.

Causal Encoder-Decoder (YOCO-style) MoE built on MiniCPM-2B.
Targets: total ~24B, Encoder(input) active ~3B, Decoder(output) active ~6B.

The attention term is an *estimate* (1.25x of MHA) until the real CSA/HCA/MLA
projection dims are fixed; replace `attn` accordingly, then re-tune the routed
expert counts (Nr_e / Nr_d) to land total exactly on 24.0B.
"""

# ---- base (MiniCPM-2B) ----
D = 2304          # hidden size
V = 122753        # vocab (tie embedding)
MOE_INT = 2048    # per-expert SwiGLU intermediate size

EMB = V * D                      # tied embedding, shared by enc input + dec LM head
EXPERT = 3 * D * MOE_INT         # one SwiGLU expert (gate, up, down)
ATTN = int(1.25 * 4 * D * D)     # per-layer self-attn (CSA/HCA) estimate
CROSS = 4 * D * D                # per-layer cross-attn (q,k,v,o), decoder only

# ---- Encoder = self-decoder (builds single global KV cache) ----
LE, NS_E, TK_E, NR_E = 16, 2, 8, 22
# ---- Decoder = cross-decoder (self-attn + cross-attn to global cache) ----
LD, NS_D, TK_D, NR_D = 24, 2, 12, 47


def b(x: float) -> str:
    return f"{x/1e9:.2f}B"


def main() -> None:
    enc_attn = ATTN * LE
    dec_attn = (ATTN + CROSS) * LD

    enc_active = EMB + enc_attn + LE * (NS_E + TK_E) * EXPERT
    dec_active = EMB + dec_attn + LD * (NS_D + TK_D) * EXPERT

    enc_total = enc_attn + LE * (NS_E + NR_E) * EXPERT
    dec_total = dec_attn + LD * (NS_D + NR_D) * EXPERT
    total = EMB + enc_total + dec_total

    print(f"embedding (tied)         : {b(EMB)}")
    print(f"expert (single)          : {EXPERT/1e6:.2f}M")
    print("-- Encoder (self-decoder) --")
    print(f"  layers={LE} experts={NS_E}+{NR_E} top_k={TK_E}")
    print(f"  ACTIVE / input token   : {b(enc_active)}")
    print(f"  stack total            : {b(enc_total)}")
    print("-- Decoder (cross-decoder) --")
    print(f"  layers={LD} experts={NS_D}+{NR_D} top_k={TK_D}")
    print(f"  ACTIVE / output token  : {b(dec_active)}")
    print(f"  stack total            : {b(dec_total)}")
    print("-- Summary --")
    print(f"  TOTAL parameters       : {b(total)}")


if __name__ == "__main__":
    main()
