import argparse
import json
from pathlib import Path

import gin
import torch

from gin_config import get_time_stamp
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--n_molecules", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    config_name = Path(args.cfg).stem
    run_name = f"{config_name}/infer_{get_time_stamp()}"
    gin.parse_config_files_and_bindings([args.cfg], bindings=[f'run_name="{run_name}"'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load objective (contains the trained policy)

    objective = gin.get_bindings("objective/gin.singleton")
    objective = gin.get_configurable("objective/gin.singleton")()
    objective.to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    objective.load_state_dict(checkpoint["model"])
    objective.eval()
    print(f"Loaded checkpoint from {args.checkpoint_path}")

    # Load the valid sampler (uses the learned forward_policy, not exploratory)
    sampler = gin.get_configurable("valid_sampler/gin.singleton")()
    sampler.policy.set_device(device)

    print(f"Sampling {args.n_molecules} molecules...")
    molecules = {}
    total = 0
    with torch.no_grad():
        for trajectories in sampler.get_trajectories_iterator(args.n_molecules, args.batch_size):
            terminal_states = trajectories.get_last_states_flat()
            reward_outputs = trajectories.get_reward_outputs()
            for i, state in enumerate(terminal_states):
                if isinstance(state, ReactionStateTerminal):
                    score = reward_outputs.proxy[i].item()
                    entry = {"score": score}
                    if reward_outputs.proxy_components is not None:
                        for name, values in reward_outputs.proxy_components.items():
                            entry[f"term_{name}"] = values[i].item()
                    molecules[state.molecule.smiles] = entry
            total += len(terminal_states)
            print(f"  Sampled {total}/{args.n_molecules}, unique so far: {len(molecules)}")

    output_path = args.output or f"inference_{config_name}_{get_time_stamp()}.json"
    with open(output_path, "w") as f:
        json.dump(molecules, f, indent=2)
    print(f"Saved {len(molecules)} unique molecules to {output_path}")
