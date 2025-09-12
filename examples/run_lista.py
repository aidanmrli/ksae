import argparse
import torch

from ksae import LISTA


def main():
	parser = argparse.ArgumentParser(description="Run LISTA on toy data")
	parser.add_argument("--input-dim", type=int, default=16)
	parser.add_argument("--code-dim", type=int, default=8)
	parser.add_argument("--steps", type=int, default=3)
	parser.add_argument("--batch", type=int, default=4)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	model = LISTA(input_dim=args.input_dim, code_dim=args.code_dim, num_iterations=args.steps)

	X = torch.randn(args.batch, args.input_dim)
	Z, hist = model(X, return_all=True)

	print(f"Input shape: {tuple(X.shape)}")
	print(f"Code shape:  {tuple(Z.shape)}")
	print(f"Iterations:  {args.steps}")
	print("First sample summary:")
	print("  B[0]:", hist["B"][0].detach().cpu().numpy())
	for t, (c, z) in enumerate(zip(hist["C"], hist["Z"][1:]), start=1):
		print(f"  t={t} C[0]:", c[0].detach().cpu().numpy())
		print(f"  t={t} Z[0]:", z[0].detach().cpu().numpy())


if __name__ == "__main__":
	main()



