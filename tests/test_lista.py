import torch
import pytest

from ksae import LISTA, soft_threshold


def manual_lista_forward(X, We, S, theta, T):
	"""
	Reference forward pass per LISTA spec using vectorized PyTorch ops.
	Supports X as (input_dim,) or (batch, input_dim).
	"""
	is_vector = X.dim() == 1
	if is_vector:
		X = X.unsqueeze(0)
	B = X @ We.t()
	Z = soft_threshold(B, theta)
	for _ in range(T):
		C = B + Z @ S.t()
		Z = soft_threshold(C, theta)
	return Z.squeeze(0) if is_vector else Z


@pytest.mark.parametrize("input_dim,code_dim,T", [(6, 4, 0), (6, 4, 1), (6, 4, 3)])
def test_lista_matches_manual_vector(input_dim, code_dim, T):
	torch.manual_seed(0)
	model = LISTA(input_dim=input_dim, code_dim=code_dim, num_iterations=T)

	# Use deterministic parameters
	with torch.no_grad():
		model.We.copy_(torch.randn(code_dim, input_dim))
		model.S.copy_(torch.randn(code_dim, code_dim) * 0.2)
		model.theta.copy_(torch.rand(code_dim) * 0.5)

	X = torch.randn(input_dim)
	Z_model = model(X)
	Z_ref = manual_lista_forward(X, model.We, model.S, model.theta, T)
	assert torch.allclose(Z_model, Z_ref, atol=0.0, rtol=0.0)


def test_lista_matches_manual_batch():
	input_dim, code_dim, T = 5, 7, 2
	torch.manual_seed(1)
	model = LISTA(input_dim=input_dim, code_dim=code_dim, num_iterations=T)
	with torch.no_grad():
		model.We.copy_(torch.randn(code_dim, input_dim))
		model.S.copy_(torch.randn(code_dim, code_dim) * 0.1)
		model.theta.copy_(torch.rand(code_dim) * 0.3)

	X = torch.randn(3, input_dim)
	Z_model = model(X)
	Z_ref = manual_lista_forward(X, model.We, model.S, model.theta, T)
	assert torch.allclose(Z_model, Z_ref, atol=0.0, rtol=0.0)


def test_lista_return_all_history_shapes():
	input_dim, code_dim, T = 4, 4, 3
	model = LISTA(input_dim=input_dim, code_dim=code_dim, num_iterations=T)
	X = torch.randn(input_dim)
	Z, hist = model(X, return_all=True)
	assert isinstance(hist, dict)
	assert "B" in hist and "C" in hist and "Z" in hist
	assert hist["B"].shape == (code_dim,)
	# Expect T entries for C^(t) and Z list contains Z^(0)..Z^(T) -> T+1 entries
	assert len(hist["C"]) == T
	assert len(hist["Z"]) == T + 1
	for c in hist["C"]:
		assert c.shape == (code_dim,)
	for z in hist["Z"]:
		assert z.shape == (code_dim,)



