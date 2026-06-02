"""
Visualise an encoder-decoder round-trip against the original terrain.

    python bench.py --ckpt runs/.../checkpoints/epoch-0500.pt
    python bench.py --size 64 --loop

Renderer setup:
    cargo run -p voxelsim-renderer --release                  # input  → 8080
    cargo run -p voxelsim-renderer --release -- --virtual 0  # output → 8082
"""

import argparse
import time

import numpy as np
import torch
import voxelsim

from representation import TerrainBatch, SimpleCNNEncoder, SimpleCNNDecoder, show_voxels


def load_model(ckpt_path, size, embedding_dim, device):
    encoder = SimpleCNNEncoder(voxel_size=size, embedding_dim=embedding_dim).to(device)
    decoder = SimpleCNNDecoder(voxel_size=size, embedding_dim=embedding_dim).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")
    encoder.eval(); decoder.eval()
    return encoder, decoder


def build_sample(size):
    g   = voxelsim.TerrainGenerator()
    cfg = voxelsim.TerrainConfig.default_py()
    cfg.set_seed_py(int(np.random.randint(0, 2**31)))
    cfg.set_world_size_py(size)
    g.generate_terrain_py(cfg)
    return TerrainBatch.world_to_voxeldata(g.generate_world_py(), size)


def run_once(encoder, decoder, client_in, client_out, size, device):
    vd = build_sample(size)
    show_voxels(vd, client_in)
    with torch.no_grad():
        out = encoder.encode([vd.to_device(device)])
        embedding = out[0] if isinstance(out, tuple) else out
        logits = decoder.decode(embedding)["logits"]
    show_voxels(logits, client_out)
    print(f"{vd.occupied_coords.shape[0]} voxels  |  input → 8080  reconstruction → 8082")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--dim",  type=int, default=1000)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_model(args.ckpt, args.size, args.dim, device)

    client_in  = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
    client_out = voxelsim.RendererClient("127.0.0.1", 8082, 8083, 8090, 9090)
    client_in.connect_py(0)
    client_out.connect_py(0)

    run_once(encoder, decoder, client_in, client_out, args.size, device)
    if args.loop:
        try:
            while True:
                time.sleep(3)
                run_once(encoder, decoder, client_in, client_out, args.size, device)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
