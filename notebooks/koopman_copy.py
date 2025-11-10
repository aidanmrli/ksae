#!/usr/bin/env python
# coding: utf-8

# # Koopman Representations

# ### Imports

# In[3]:


import os
from typing import Optional, Callable, Any, Sequence, Mapping
from datetime import datetime
from abc import ABC, abstractmethod
import functools
import importlib
import shutil

# !pip install ml-collections
import ml_collections
import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

clear_output()


# ### Types

# In[ ]:


Params = Any
PRNGKey = Any
Unused = Any
Metrics = Mapping[str, jnp.ndarray]
key: PRNGKey = jax.random.PRNGKey(0)


# ### Configuration

# In[ ]:


def get_config():
  _C = ml_collections.ConfigDict()
  _C.SEED = 0

  _C.ENV = ml_collections.ConfigDict()
  _C.ENV.ENV_NAME = "DuffingOscillator"   # from ["DuffingOscillator", "DS239"]

  _C.ENV.DS239 = ml_collections.ConfigDict()
  _C.ENV.DS239.LAMBDA = -1.0
  _C.ENV.DS239.MU = -0.1
  _C.ENV.DS239.DT = 0.1

  _C.ENV.DUFFING = ml_collections.ConfigDict()
  _C.ENV.DUFFING.DT = 0.01

  _C.MODEL = ml_collections.ConfigDict()
  _C.MODEL.MODEL_NAME = "SparseKM"   # from ["GenericKM", "SparseKM", "LISTAKM"]
  _C.MODEL.NORM_FN = "id"   # from ["id", "ball"]
  _C.MODEL.TARGET_SIZE = 16 # zdim

  # loss coefficients
  _C.MODEL.RES_COEFF = 1.0
  _C.MODEL.RECONST_COEFF = 0.02
  _C.MODEL.PRED_COEFF = 0.
  _C.MODEL.SPARSITY_COEFF = 1.0

  _C.MODEL.ENCODER = ml_collections.ConfigDict()
  _C.MODEL.ENCODER.LAYERS = [16, 16]
  _C.MODEL.ENCODER.LAST_RELU = False
  _C.MODEL.ENCODER.USE_BIAS = False

  _C.MODEL.ENCODER.LISTA = ml_collections.ConfigDict()
  _C.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
  _C.MODEL.ENCODER.LISTA.L = 1e3
  _C.MODEL.ENCODER.LISTA.ALPHA = 0.1
  _C.MODEL.ENCODER.LISTA.LINEAR_ENCODER = False

  _C.MODEL.DECODER = ml_collections.ConfigDict()
  _C.MODEL.DECODER.LAYERS = []  # linear decoder should be good
  _C.MODEL.DECODER.USE_BIAS = False

  _C.TRAIN = ml_collections.ConfigDict()
  _C.TRAIN.NUM_STEPS = 2_000
  _C.TRAIN.BATCH_SIZE = 256
  _C.TRAIN.DATA_SIZE = 256 * 8
  _C.TRAIN.LR = 1e-3

  return _C

cfg = get_config()


# ## Environment

# ### Env Common

# In[ ]:


class Env(ABC):
  """Base Env class."""

  def __init__(self, cfg: Optional[ml_collections.ConfigDict], *args, **kwargs):
    self.cfg = cfg

  @abstractmethod
  def reset(self, rng=jnp.ndarray) -> jnp.ndarray:
    pass

  @abstractmethod
  def step(self, state: jnp.ndarray, action: Optional[jnp.ndarray] = None, *args) -> jnp.ndarray:
    pass

  @property
  def observation_size(self) -> int:
    rng = jax.random.PRNGKey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.shape[-1]

  @property
  def action_size(self) -> int:
    return self.action_size

  @property
  def unwrapped(self) -> 'Env':
    return self


class Wrapper(Env):
  """Base Wrapper class."""

  def __init__(self, env: Env):
    super().__init__(cfg=None)
    self.env = env

  def reset(self, rng: jnp.ndarray) -> jnp.ndarray:
    return self.env.reset(rng)

  def step(self, state: jnp.ndarray, action: Optional[jnp.ndarray] = None, *args) -> jnp.ndarray:
    return self.env.step(state, action, *args)

  @property
  def observation_size(self) -> int:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Env:
    return self.env.unwrapped


# In[ ]:


class VectorWrapper(Wrapper):
  """Class for vectorization of env ops."""

  def __init__(self, env: Env, batch_size: int):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jnp.ndarray) -> jnp.ndarray:
    rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: jnp.ndarray, action: Optional[jnp.ndarray] = None, *args) -> jnp.ndarray:
    return jax.vmap(self.env.step, in_axes=(0, 0))(state, action, *args)


# In[ ]:


def generate_trajectory(
    env_step: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray],
    init_state: Optional[jnp.ndarray],
    length: Optional[int],
    rng: PRNGKey,
    actions: Optional[jnp.ndarray]=None) -> jnp.ndarray:

    if actions is None:
      assert length is not None
      def scan_fn(carry, action):
        state = carry
        nstate = env_step(state)
        return nstate, nstate
      _, states = jax.lax.scan(scan_fn, init_state, (), length)
    else:
      def scan_fn(carry, action):
        state = carry
        nstate = env_step(state, action)
        return nstate, nstate
      _, states = jax.lax.scan(scan_fn, init_state, actions)

    # states = jnp.concatenate([init_state[None, :], states], axis=0)
    return states


# In[ ]:


#TODO(mahanf): add other integrators (RK4, Heun, etc)

def integrate_euler(x: jnp.ndarray, u: Optional[jnp.ndarray], dt: float,
    dynamics_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]) -> jnp.ndarray:
    return x + dt * dynamics_fn(x, u)


# ### DS239

# In[ ]:


class DS239(Env):

  def __init__(self, cfg: ml_collections.ConfigDict):
    super().__init__(cfg)
    self.const_lambda = cfg.ENV.DS239.LAMBDA
    self.const_mu = cfg.ENV.DS239.MU
    self.dt = cfg.ENV.DS239.DT

  @property
  def action_size(self) -> int:
    return 0

  def reset(self, rng: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(rng, shape=[2, ], minval=-1.0, maxval=1.0)

  def step(self, state: jnp.ndarray, action: Optional[jnp.ndarray] = None, *args) -> jnp.ndarray:

    @jax.jit
    def dynamics_fn(state, action=None):
      x1, x2 = state
      xd1 = self.const_mu * x1
      xd2 = self.const_lambda * (x2 - x1 ** 2)
      return jnp.array([xd1, xd2])

    return integrate_euler(state, None, self.dt, dynamics_fn)


# ### Duffing Oscillator

# In[ ]:


class DuffingOscillator(Env):

  def __init__(self, cfg: ml_collections.ConfigDict):
    super().__init__(cfg)
    self.dt = cfg.ENV.DUFFING.DT

  @property
  def action_size(self) -> int:
    return 0

  def reset(self, rng: jnp.ndarray) -> jnp.ndarray:
    rng_x1, rng_x2 = jax.random.split(rng)
    x1 = jax.random.uniform(rng_x1, minval=-1.5, maxval=1.5)
    x2 = jax.random.uniform(rng_x2, minval=-1.0, maxval=1.0)
    return jnp.array([x1, x2])

  def step(self, state: jnp.ndarray, action: Optional[jnp.ndarray] = None, *args) -> jnp.ndarray:

    @jax.jit
    def dynamics_fn(state, action=None):
      x1, x2 = state
      xd1 = x2
      xd2 = x1 - x1 ** 3
      return jnp.array([xd1, xd2])

    return integrate_euler(state, None, self.dt, dynamics_fn)


# ### Misc

# In[ ]:


_ENV_REGISTRY = {
    "DS239": DS239,
    "DuffingOscillator": DuffingOscillator,
}

def make_env(cfg: ml_collections.ConfigDict):
  return _ENV_REGISTRY[cfg.ENV.ENV_NAME](cfg)


# In[ ]:


key, key_traj, key_init = jax.random.split(key, 3)
env = make_env(cfg)
env = VectorWrapper(env, 256)
init_state = env.reset(key_init)
traj = generate_trajectory(env.step, init_state, 200, key_traj)
plt.plot(traj[:, :, 0], traj[:, :, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('{\u03BC x1, \u03BB (x2 - x1 ^ 2)}')
plt.show()


# ## Model

# ### Model Basics

# In[ ]:


class MLPcoder(nn.Module):
  target_size: int
  hidden_layers: Sequence[int]
  last_relu: bool = False
  use_bias: bool = False

  def setup(self):
    last_act = nn.relu if self.last_relu else lambda x: x
    self.coder = nn.Sequential([
        entry
        for layer in [[nn.Dense(hidden, use_bias=True), nn.relu] for hidden in self.hidden_layers]
        for entry in layer
        ] + [nn.Dense(self.target_size, use_bias=True), last_act])

  def __call__(self, input):
    return self.coder(input)


# In[ ]:


def shrink(x: jnp.ndarray, threshold: float) -> jnp.ndarray:
  return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, jnp.zeros_like(x))

def _fori_loop(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val

class LISTA(nn.Module):
  cfg: ml_collections.ConfigDict
  xdim: int
  Wd_init: jnp.ndarray # trainable for now
  dtype: str = "float32"

  @staticmethod
  def _get_S_initializer(Wd: jnp.ndarray, L):
    """S mat initializer."""
    def _S_initializer(
      key: PRNGKey,
      shape: jnp.ndarray,
      dtype: Unused = "float32",
      ) -> jnp.ndarray:
      del key, shape, dtype
      zdim = Wd.shape[0]
      return jnp.eye(zdim) - (1. / L) * Wd @ Wd.T
    return _S_initializer

  @staticmethod
  def _get_We_initializer(Wd: jnp.ndarray, L):
    """We mat initializer."""
    def _We_initializer(
      key: PRNGKey,
      shape: jnp.ndarray,
      dtype: Unused = "float32",
      ) -> jnp.ndarray:
      del key, shape, dtype
      return (1. / L) * Wd.T
    return _We_initializer

  def setup(self):

    # # init unnormalized Wd
    # self._Wd = self.param(
    #     "Wd", nn.initializers.normal(), (self.zdim, self.xdim), self.dtype)

    xdim = self.xdim
    zdim = self.cfg.MODEL.TARGET_SIZE
    n_loops = self.cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
    alpha = self.cfg.MODEL.ENCODER.LISTA.ALPHA
    L = self.cfg.MODEL.ENCODER.LISTA.L
    use_linear_encode = self.cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER

    assert self.Wd_init.shape == (zdim, xdim)

    if use_linear_encode:
      self.We = nn.Dense(
          zdim, use_bias=False, kernel_init=self._get_We_initializer(self.Wd_init, L))
    else:
      self.We = MLPcoder(
          target_size=zdim,
          hidden_layers=self.cfg.MODEL.ENCODER.LAYERS,
          use_bias=self.cfg.MODEL.ENCODER.USE_BIAS,
          last_relu=self.cfg.MODEL.ENCODER.LAST_RELU)

    # self.We = self.param(
    #     "We", self._get_We_initializer(self.Wd_init), (self.xdim, self.zdim), self.dtype)

    self.S = self.param(
        "S", self._get_S_initializer(self.Wd_init, L), (zdim, zdim), self.dtype)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # fori_loop = jax.lax.fori_loop
    fori_loop = _fori_loop

    n_loops = self.cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
    alpha = self.cfg.MODEL.ENCODER.LISTA.ALPHA
    L = self.cfg.MODEL.ENCODER.LISTA.L

    nonsparse_code = self.We(x)
    return fori_loop(
        0, n_loops,
        lambda _, z: shrink(z @ self.S + nonsparse_code, alpha/L),
        shrink(nonsparse_code, alpha/L))


# In[ ]:


class KoopmanMachine(ABC):

  # TODO(mahanf): support actuated envs?
  action_size: Optional[int] = None

  def __init__(self,
               cfg: ml_collections.ConfigDict,
               observation_size: int,
               ):

    self.cfg = cfg
    self.target_size = cfg.MODEL.TARGET_SIZE
    self.observation_size = observation_size

    self.loss_grad = jax.value_and_grad(self.loss, has_aux=True)


  @abstractmethod
  def init(self, key: PRNGKey) -> Params:
    pass

  @abstractmethod
  def encode(self, params: Params, x: jnp.array):
    """Representations are extracted here."""
    pass

  @abstractmethod
  def decode(self, params: Params, y: jnp.array):
    """Takes representations from encoded space to real space.
    This could be only for monitoring purposes.
    """
    pass

  @abstractmethod
  def kmatrix(self, params: Params):
    """Koopman Operator Matrix.
      Extracts the learned kmatrix from parameters.
    """
    pass

  @abstractmethod
  def loss(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray) -> (jnp.ndarray, Metrics):
    """Contains the bread and butter of the method."""

  def update(self, params: Params, x: Optional[jnp.ndarray]=None, nx: Optional[jnp.ndarray]=None):
    """Called every so often during training."""
    pass

  def step_latent(self, params: Params, y: jnp.ndarray) -> jnp.ndarray:
    kmatrix = self.kmatrix(params)
    ny = y @ kmatrix
    return ny

  def step_env(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
    y = self.encode(params, x)
    ny = self.step_latent(params, y)
    nx = self.decode(params, ny)
    return nx


# In[ ]:


def norm_wrapper(cls):
  class WrapperClass(cls):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def encode(self, params: Params, x: jnp.array) -> jnp.ndarray:
      y = super().encode(params, x)
      return super().norm_fn(y)

    def step_latent(self, params: Params, y: jnp.ndarray) -> jnp.ndarray:
      ny = super().step_latent(params, y)
      return super().norm_fn(ny)
  return WrapperClass


# In[ ]:


# eigvals cannot be computed on gpu :)
eigvals_fn = jax.jit(jnp.linalg.eigvals, backend='cpu')


# ### Reconstructional Models

# #### Reconstruction w/ Explicit Linear Alignment

# In[ ]:


@norm_wrapper
class GenericKM(KoopmanMachine):

  def __init__(self,
               cfg: ml_collections.ConfigDict,
               observation_size: int,
               ):

    super().__init__(cfg, observation_size)

    self._encoder = MLPcoder(
        target_size=cfg.MODEL.TARGET_SIZE,
        hidden_layers=cfg.MODEL.ENCODER.LAYERS,
        use_bias=cfg.MODEL.ENCODER.USE_BIAS,
        last_relu=cfg.MODEL.ENCODER.LAST_RELU,
    )
    self._decoder = MLPcoder(
        target_size=observation_size,
        hidden_layers=cfg.MODEL.DECODER.LAYERS,
        use_bias=cfg.MODEL.DECODER.USE_BIAS,
        last_relu=False,
    )


  def norm_fn(self, x: jnp.ndarray) -> jnp.ndarray:
    norm_fn = {
        'id': lambda x: x,
        'ball': lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True),
    }[cfg.MODEL.NORM_FN]
    return norm_fn(x)

  def _init_kmat(self, rng: PRNGKey) -> jnp.ndarray:
    return jnp.eye(self.target_size)

  def init(self, key: PRNGKey) -> Params:
    key, key_kmat, key_encoder, key_decoder = jax.random.split(key, 4)
    dummy_state = jnp.zeros([self.observation_size])
    dummy_latent = jnp.zeros([self.target_size])
    encoder_params = self._encoder.init(key_encoder, dummy_state)
    decoder_params = self._decoder.init(key_decoder, dummy_latent)
    kmat_params = self._init_kmat(key_kmat)
    return {
        'encoder': encoder_params,
        'decoder': decoder_params,
        'kmat': kmat_params,
    }

  def encode(self, params: Params, x: jnp.array):
    return self._encoder.apply(params['encoder'], x)

  def decode(self, params: Params, y: jnp.array):
    return self._decoder.apply(params['decoder'], y)

  def kmatrix(self, params: Params, x: Optional[jnp.ndarray]=None, nx: Optional[jnp.ndarray]=None):
    return params['kmat']

  def residual(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    """Determines how linearly aligned x & nx are in the latent space."""

    y = self.encode(params, x)
    ny = self.encode(params, nx)
    kmatrix = self.kmatrix(params)
    return jnp.linalg.norm(y @ kmatrix - ny, axis=-1)

  def reconstruction(self, params: Params, x: jnp.ndarray):
    return self.decode(params, self.encode(params, x))

  def sparsity_loss(self, params: Params, x: jnp.ndarray):
    z = self.encode(params, x)
    return jnp.linalg.norm(z, axis=-1, ord=1).mean()

  def loss(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):

    # linear prediction loss
    kmat = self.kmatrix(params)
    prediction = self.decode(params, self.encode(params, x) @ kmat)
    prediction_loss = jnp.linalg.norm(prediction - nx, axis=-1).mean()

    # linear dynamics loss
    residual_loss = self.residual(params, x, nx).mean()

    # reconstruction loss
    reconst_loss = jnp.linalg.norm(x - self.reconstruction(params, x), axis=-1).mean()
    reconst_loss += jnp.linalg.norm(nx - self.reconstruction(params, nx), axis=-1).mean()

    # sparsity loss
    sparsity_loss = self.sparsity_loss(params, x)
    sparsity_loss += self.sparsity_loss(params, nx)
    sparsity_loss *= 0.5

    # koopman matrix eigenvalues
    max_eigenvalue = jnp.max(jnp.real(eigvals_fn(kmat)))

    # nonzero codes
    num_nonzero_codes = jnp.linalg.norm(self.encode(params, x), axis=-1, ord=0).mean()
    sparsity_ratio = 1. - num_nonzero_codes / self.target_size

    loss = (
        cfg.MODEL.RES_COEFF * residual_loss
        + cfg.MODEL.RECONST_COEFF * reconst_loss
        + cfg.MODEL.PRED_COEFF * prediction_loss
        + cfg.MODEL.SPARSITY_COEFF * sparsity_loss)

    metrics = {
        'loss': loss,
        'residual_loss': residual_loss,
        'reconst_loss': reconst_loss,
        'prediction_loss': prediction_loss,
        'A_max_eigenvalue': max_eigenvalue,
        'sparsity_ratio': sparsity_ratio,
    }

    return loss, metrics


# #### LISTA Koopman Machine

# In[ ]:


class LISTAKM(KoopmanMachine):
  def __init__(self,
               cfg: ml_collections.ConfigDict,
               observation_size: int,
               ):
    super().__init__(cfg, observation_size)


  def _init_kmat(self, rng: PRNGKey) -> jnp.ndarray:
    return jnp.eye(self.target_size)

  def _init_dict(self, rng: PRNGKey) -> jnp.ndarray:
    return jax.random.normal(rng, shape=(self.target_size, self.observation_size))

  def init(self, key: PRNGKey) -> Params:
    key, key_kmat, key_lista, key_dict = jax.random.split(key, 4)
    dict_params = self._init_dict(key_dict)
    self._lista = LISTA(
      cfg=self.cfg,
      xdim=self.observation_size,
      Wd_init=dict_params,
    )
    dummy_state = jnp.zeros([self.observation_size])
    lista_params = self._lista.init(key_lista, dummy_state)
    kmat_params = self._init_kmat(key_kmat)
    return {
        'lista': lista_params,
        'dict': dict_params,
        'kmat': kmat_params,
    }

  def encode(self, params: Params, x: jnp.array):
    return self._lista.apply(params['lista'], x)

  def decode(self, params: Params, y: jnp.array):
    wd = params['dict']
    wd /= jnp.maximum(jnp.linalg.norm(wd, axis=1, keepdims=True), 1e-4)
    return y @ wd

  def kmatrix(self, params: Params, x: Optional[jnp.ndarray]=None, nx: Optional[jnp.ndarray]=None):
    return params['kmat']

  def residual(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    """Determines how linearly aligned x & nx are in the latent space."""

    y = self.encode(params, x)
    ny = self.encode(params, nx)
    kmatrix = self.kmatrix(params)
    return jnp.linalg.norm(y @ kmatrix - ny, axis=-1)

  def reconstruction(self, params: Params, x: jnp.ndarray):
    return self.decode(params, self.encode(params, x))

  def sparsity_loss(self, params: Params, x: jnp.ndarray):
    z = self.encode(params, x)
    return self.cfg.MODEL.ENCODER.LISTA.ALPHA * jnp.linalg.norm(z, axis=-1, ord=1).mean()

  def loss(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):

    # linear prediction loss
    kmat = self.kmatrix(params)
    prediction = self.decode(params, self.encode(params, x) @ kmat)
    prediction_loss = jnp.linalg.norm(prediction - nx, axis=-1).mean()

    # linear dynamics loss
    residual_loss = self.residual(params, x, nx).mean()

    # reconstruction loss
    reconst_loss = jnp.linalg.norm(x - self.reconstruction(params, x), axis=-1).mean()
    reconst_loss += jnp.linalg.norm(nx - self.reconstruction(params, nx), axis=-1).mean()

    # sparsity loss
    sparsity_loss = self.sparsity_loss(params, x)
    sparsity_loss += self.sparsity_loss(params, nx)
    sparsity_loss *= 0.5

    # koopman matrix eigenvalues
    max_eigenvalue = jnp.max(jnp.real(eigvals_fn(kmat)))

    # nonzero codes
    num_nonzero_codes = jnp.linalg.norm(self.encode(params, x), axis=-1, ord=0).mean()
    sparsity_ratio = 1. - num_nonzero_codes / self.target_size

    loss = (
        cfg.MODEL.RES_COEFF * residual_loss
        + cfg.MODEL.RECONST_COEFF * reconst_loss
        + cfg.MODEL.PRED_COEFF * prediction_loss
        + cfg.MODEL.SPARSITY_COEFF * sparsity_loss)


    metrics = {
        'loss': loss,
        'residual_loss': residual_loss,
        'reconst_loss': reconst_loss,
        'sparsity_loss': sparsity_loss,
        'prediction_loss': prediction_loss,
        'sparsity_ratio': sparsity_ratio,
        'A_max_eigenvalue': max_eigenvalue,
    }

    return loss, metrics


# #### [DEPRECATED] Reconstruction w/ Implicit Linear Alignment

# In[ ]:


class ReconKMImplicit(KoopmanMachine):

  def __init__(self,
               cfg: ml_collections.ConfigDict,
               observation_size: int,
               ):

    super().__init__(cfg, observation_size)

    self._encoder = MLPcoder(
        target_size=cfg.MODEL.TARGET_SIZE,
        hidden_layers=cfg.MODEL.ENCODER.LAYERS,
        use_bias=cfg.MODEL.ENCODER.USE_BIAS,
        last_relu=cfg.MODEL.ENCODER.LAST_RELU,
    )
    self._decoder = MLPcoder(
        target_size=observation_size,
        hidden_layers=cfg.MODEL.DECODER.LAYERS,
        use_bias=cfg.MODEL.DECODER.USE_BIAS,
        last_relu=False,
    )

  def init(self, key: PRNGKey) -> Params:
    key, key_encoder, key_decoder = jax.random.split(key, 3)
    dummy_state = jnp.zeros([self.observation_size])
    dummy_latent = jnp.zeros([self.target_size])
    encoder_params = self._encoder.init(key_encoder, dummy_state)
    decoder_params = self._decoder.init(key_decoder, dummy_latent)
    return {'encoder': encoder_params, 'decoder': decoder_params}

  def encode(self, params: Params, x: jnp.array):
    return self._encoder.apply(params['encoder'], x)

  def decode(self, params: Params, y: jnp.array):
    return self._decoder.apply(params['decoder'], y)

  def kmatrix(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    y = self.encode(params, x)
    ny = self.encode(params, nx)
    return jnp.linalg.lstsq(y, ny)[0]

  def residual(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    """Determines how linearly aligned x & nx are in the latent space."""

    y = self.encode(params, x)
    ny = self.encode(params, nx)
    return jnp.linalg.lstsq(y, ny)[1] # kmatrix, residual

  def reconstruction(self, params: Params, x: jnp.ndarray):
    return self.decode(params, self.encode(params, x))

  def loss(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):

    # linear dynamics loss
    residual_loss = self.residual(params, x, nx).mean()

    # reconstruction loss
    reconst_loss = jnp.linalg.norm(x - self.reconstruction(params, x), axis=-1).mean()
    reconst_loss += jnp.linalg.norm(nx - self.reconstruction(params, nx), axis=-1).mean()
    reconst_loss *= 0.02

    loss = residual_loss + reconst_loss

    return loss, {
        'loss': loss,
        'residual_loss': residual_loss,
        'reconst_loss': reconst_loss,
    }


# #### [DEPRECATED] SimSiam Implicit Model

# In[ ]:


class SimSiamKMImplicit(KoopmanMachine):

  def __init__(self,
               cfg: ml_collections.ConfigDict,
               observation_size: int,
               ):

    super().__init__(cfg, observation_size)

    self._encoder = MLPcoder(
        target_size=cfg.MODEL.TARGET_SIZE,
        hidden_layers=cfg.MODEL.ENCODER.LAYERS,
        use_bias=cfg.MODEL.ENCODER.USE_BIAS,
        last_relu=cfg.MODEL.ENCODER.LAST_RELU,
    )

    self._predictor = MLPcoder(
        target_size=cfg.MODEL.TARGET_SIZE,
        hidden_layers=[cfg.MODEL.TARGET_SIZE],
        use_bias=False,
        last_relu=False,
    )

    self._decoder = MLPcoder(
        target_size=observation_size,
        hidden_layers=cfg.MODEL.DECODER.LAYERS,
        use_bias=cfg.MODEL.DECODER.USE_BIAS,
        last_relu=False,
    )

  def init(self, key: PRNGKey) -> Params:
    key, key_encoder, key_decoder = jax.random.split(key, 3)
    dummy_state = jnp.zeros([self.observation_size])
    dummy_latent = jnp.zeros([self.target_size])
    encoder_params = self._encoder.init(key_encoder, dummy_state)
    decoder_params = self._decoder.init(key_decoder, dummy_latent)
    return {'encoder': encoder_params, 'decoder': decoder_params}

  def encode(self, params: Params, x: jnp.array):
    return self._encoder.apply(params['encoder'], x)

  def decode(self, params: Params, y: jnp.array):
    return self._decoder.apply(params['decoder'], y)

  def kmatrix(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    y = self.encode(params, x)
    ny = self.encode(params, nx)
    return jnp.linalg.lstsq(y, ny)[0]

  def residual(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):
    """Determines how linearly aligned x & nx are in the latent space."""

    y = self.encode(params, x)
    ny = jax.lax.stop_gradient(self.encode(params, nx))
    return jnp.linalg.lstsq(y, ny)[1]

  def reconstruction(self, params: Params, x: jnp.ndarray):
    return self.decode(params, jax.lax.stop_gradient(self.encode(params, x)))

  def loss(self, params: Params, x: jnp.ndarray, nx: jnp.ndarray):

    # linear dynamics loss
    residual_loss = self.residual(params, x, nx).mean()
    residual_loss += self.residual(params, nx, x).mean()

    # NOTE: reconstruction is only for visualization purposes
    # reconstruction loss
    reconst_loss = jnp.linalg.norm(x - self.reconstruction(params, x), axis=-1).mean()
    reconst_loss += jnp.linalg.norm(nx - self.reconstruction(params, nx), axis=-1).mean()
    reconst_loss *= 0.02

    loss = residual_loss + reconst_loss

    return loss, {
        'loss': loss,
        'residual_loss': residual_loss,
        'reconst_loss': reconst_loss,
    }


# ### Model Common

# In[ ]:


_KM_REGISTRY = {
    "GenericKM": GenericKM,
    "LISTAKM": LISTAKM,
    # "ReconKMImplicit": ReconKMImplicit,
    # "SimSiamKMImplicit": SimSiamKMImplicit,
}

def make_koopman_machine(cfg: ml_collections.ConfigDict, observation_size: int):
  return _KM_REGISTRY[cfg.MODEL.MODEL_NAME](cfg, observation_size)


# In[ ]:


def make_km_env_n_step(
    params: Params,
    km: KoopmanMachine,
    x: jnp.ndarray,
    length: int,
    reencode_at_every: int = 1,
    ):
    """
    env_step fn takes in a batch of states and outputs the N next states

    reencode_at_every:  int pretty self-explanatory
    """


    if reencode_at_every == 1:
      # km_step_env = functools.partial(km.step_env, params=params)
      km_step_env = lambda x: km.step_env(params, x)
      km_traj = generate_trajectory(km_step_env, x, length, key)
      return km_traj

    elif reencode_at_every == 0:
      # km_latent_step = functools.partial(km.step_latent, params=params)
      km_latent_step = lambda z: km.step_latent(params, z)
      y = km.encode(params, x)
      km_latent_traj = generate_trajectory(km_latent_step, y, length, key)
      km_traj = jax.vmap(km.decode, (None, 0))(params, km_latent_traj)
      return km_traj

    else:
      assert length % reencode_at_every == 0
      num_slices = length // reencode_at_every
      km_trajs = []
      # km_latent_step = functools.partial(km.step_latent, params=params)
      km_latent_step = lambda z: km.step_latent(params, z)
      for _ in range(num_slices):
        y = km.encode(params, x)
        km_latent_traj = generate_trajectory(km_latent_step, y, reencode_at_every, key)
        decoded = jax.vmap(km.decode, (None, 0))(params, km_latent_traj)
        km_trajs.append(decoded)
        x = decoded[-1, :, :]
      km_traj = jnp.concatenate(km_trajs, axis=0)
      return km_traj

def plot_eval(params, km, x, length=100, env_stp_fn=None) -> None:
  clear_output(wait=True)

  # ------------------------------
  #           phase plots        #
  # ------------------------------
  # use predefined reencoding freqs
  km_traj_0 = make_km_env_n_step(params, km, x, length, reencode_at_every=0)
  km_traj_1 = make_km_env_n_step(params, km, x, length, reencode_at_every=1)
  km_traj_10 = make_km_env_n_step(params, km, x, length, reencode_at_every=10)
  km_traj_25 = make_km_env_n_step(params, km, x, length, reencode_at_every=25)
  km_traj_50 = make_km_env_n_step(params, km, x, length, reencode_at_every=50)
  fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(40, 20))
  axs[0, 0].plot(km_traj_0[:, :, 0], km_traj_0[:, :, 1])
  axs[0, 0].set_title('reencode [x]')
  axs[0, 1].plot(km_traj_1[:, :, 0], km_traj_1[:, :, 1])
  axs[0, 1].set_title('reencode @ 1')
  axs[0, 2].plot(km_traj_10[:, :, 0], km_traj_10[:, :, 1])
  axs[0, 2].set_title('reencode @ 10')
  axs[0, 3].plot(km_traj_25[:, :, 0], km_traj_25[:, :, 1])
  axs[0, 3].set_title('reencode @ 25')
  axs[0, 4].plot(km_traj_50[:, :, 0], km_traj_50[:, :, 1])
  axs[0, 4].set_title('reencode @ 50')
  for ax in axs.flat:
    ax.set(xlabel='x1', ylabel='x2')

  # ------------------------------
  #         L2 error plots       #
  # ------------------------------
  # get ground truth trajectory
  gt_traj = generate_trajectory(env_stp_fn, x, length, key)
  err_fn = lambda x, y: jnp.linalg.norm(y - x, axis=-1).mean(axis=1)
  err_0 = err_fn(gt_traj, km_traj_0)
  err_1 = err_fn(gt_traj, km_traj_1)
  err_10 = err_fn(gt_traj, km_traj_10)
  err_25 = err_fn(gt_traj, km_traj_25)
  err_50 = err_fn(gt_traj, km_traj_50)
  axs[1, 0].plot(err_0)
  axs[1, 1].plot(err_1)
  axs[1, 2].plot(err_10)
  axs[1, 3].plot(err_25)
  axs[1, 4].plot(err_50)
  axs[2, 0].plot(err_0, label="reencode @ 0")
  axs[2, 0].plot(err_1, label="reencode @ 1")
  axs[2, 0].plot(err_10, label="reencode @ 10")
  axs[2, 0].plot(err_25, label="reencode @ 25")
  axs[2, 0].plot(err_50, label="reencode @ 50")
  axs[2, 0].set_title('L2 error by horizon')
  axs[2, 1].axis('off')
  axs[2, 2].axis('off')
  axs[2, 3].axis('off')
  axs[2, 4].axis('off')
  axs[2, 0].legend()
  axs[2, 0].colspan = 5

  plt.tight_layout()
  plt.legend()
  plt.show()


# ## Train

# ### Tensorboard

# In[ ]:


log_dir = './log/'
get_ipython().system('rm -rf ./log/')
if not os.path.exists(log_dir):
  os.mkdir(log_dir)
else:
  shutil.rmtree(log_dir)
  os.mkdir(log_dir)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir=./log/ --port=0')


# ### Train

# In[ ]:


cfg = get_config() # get default config
cfg.TRAIN.NUM_STEPS = 20_000
cfg.ENV.ENV_NAME = "DuffingOscillator"
# cfg.ENV.ENV_NAME = "DS239"


#@markdown SELECT TRAINING CONFIG
# ------------------------------ ## ------------------------------ #
#                           C O N F I G                            #
# ------------------------------ ## ------------------------------ #
TRAIN_CFG = 'generic_sparse' #@param ["generic", "generic_sparse", "generic_prediction", "LISTA", "LISTA_nonlinear"]


if TRAIN_CFG == 'generic':
  # ------------------------ ## -------------------------- #
  #                     G E N E R A L                       #
  # ------------------------ ## -------------------------- #
  cfg.TRAIN.LR = 1e-3
  cfg.MODEL.MODEL_NAME = "GenericKM"
  cfg.MODEL.TARGET_SIZE = 64
  cfg.MODEL.NORM_FN = 'id'
  cfg.MODEL.DECODER.LAYERS = []

  # misc.
  # cfg.MODEL.DECODER.USE_BIAS = True
  # cfg.TRAIN.BATCH_SIZE = 256
  # cfg.TRAIN.DATA_SIZE = cfg.TRAIN.BATCH_SIZE * 10
  # cfg.TRAIN.BATCH_SIZE = 512
  # cfg.MODEL.ENCODER.LAYERS = [64, 64, 64]
  # cfg.MODEL.ENCODER.LAST_RELU = True
  # cfg.MODEL.ENCODER.USE_BIAS = True

  # bigger encoder
  cfg.MODEL.ENCODER.LAYERS = [64, 64]

  cfg.MODEL.SPARSITY_COEFF = 0.0

elif TRAIN_CFG == 'generic_sparse':
  # ------------------------ ## -------------------------- #
  #                     G E N E R A L                       #
  # ------------------------ ## -------------------------- #
  cfg.TRAIN.LR = 1e-3
  cfg.MODEL.MODEL_NAME = "GenericKM"
  cfg.MODEL.TARGET_SIZE = 64
  cfg.MODEL.NORM_FN = 'id'
  cfg.MODEL.DECODER.LAYERS = []

  # bigger encoder
  cfg.MODEL.ENCODER.LAYERS = [64, 64]

  cfg.MODEL.ENCODER.LAST_RELU = True
  cfg.MODEL.ENCODER.USE_BIAS = True

  cfg.MODEL.RECONST_COEFF = 0.5
  cfg.MODEL.SPARSITY_COEFF = .01

elif TRAIN_CFG == 'generic_prediction':
  # ------------------------ ## -------------------------- #
  #                      PREDICTION                        #
  # ------------------------ ## -------------------------- #
  cfg.MODEL.MODEL_NAME = "GenericKM"
  cfg.TRAIN.LR = 1e-3
  cfg.MODEL.DECODER.LAYERS = []

  # loss hparams
  cfg.MODEL.PRED_COEFF = 1.
  cfg.MODEL.RES_COEFF = .0
  cfg.MODEL.RECONST_COEFF = 0.0
  cfg.MODEL.SPARSITY_COEFF = .0


elif TRAIN_CFG == 'LISTA':
  # ------------------------ ## -------------------------- #
  #                     S P A R S E                        #
  # ------------------------ ## -------------------------- #
  cfg.MODEL.MODEL_NAME = "LISTAKM"
  cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
  cfg.TRAIN.LR = 1e-4
  cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
  cfg.MODEL.TARGET_SIZE = 1024 * 2
  cfg.MODEL.RES_COEFF = 1.0
  cfg.MODEL.RECONST_COEFF = 1.0
  cfg.MODEL.PRED_COEFF = .0
  cfg.MODEL.SPARSITY_COEFF = 1.0
  cfg.MODEL.NORM_FN = 'id'
  cfg.MODEL.ENCODER.LISTA.L = 1e4
  cfg.MODEL.ENCODER.LISTA.ALPHA = 0.1

elif TRAIN_CFG == 'LISTA_nonlinear':
  # ------------------------ ## -------------------------- #
  #                     S P A R S E                        #
  # ------------------------ ## -------------------------- #
  cfg.MODEL.MODEL_NAME = "LISTAKM"
  cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = False
  cfg.MODEL.ENCODER.LAYERS = [64, 64, 64]
  cfg.TRAIN.LR = 1e-4
  cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
  cfg.MODEL.TARGET_SIZE = 1024 * 2
  cfg.MODEL.RES_COEFF = 1.0
  cfg.MODEL.RECONST_COEFF = 1.0
  cfg.MODEL.PRED_COEFF = .0
  cfg.MODEL.SPARSITY_COEFF = 1.0
  cfg.MODEL.NORM_FN = 'id'
  cfg.MODEL.ENCODER.LISTA.L = 1e4
  cfg.MODEL.ENCODER.LISTA.ALPHA = 1.0
  cfg.MODEL.ENCODER.LAST_RELU = True
  cfg.MODEL.ENCODER.USE_BIAS = True


print(cfg)


# In[ ]:


# ------------------------------ ## ------------------------------ #
#
#                          S T A T E F U L                         #
#
# ------------------------------ ## ------------------------------ #
try:
  sw.close()
except:
  pass

sub_dir = str(datetime.now())
sw = jaxboard.SummaryWriter(os.path.join(log_dir, sub_dir))

seed = cfg.SEED
batch_size = cfg.TRAIN.BATCH_SIZE
num_steps = cfg.TRAIN.NUM_STEPS
lr = cfg.TRAIN.LR

env = make_env(cfg)
env = VectorWrapper(env, batch_size)
km = make_koopman_machine(cfg, env.observation_size)

key = jax.random.PRNGKey(seed)
key, key_train, key_init = jax.random.split(key, 3)
num_batches = cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE
keys_train = jax.random.split(key, num_batches)

init_params = km.init(key_init)
optimizer = optax.adam(learning_rate=lr)

train_state = TrainState.create(apply_fn=km.loss_grad, params=init_params, tx=optimizer)
# ------------------------------ ## ------------------------------ #


# In[ ]:


# ------------------------------ ## ------------------------------ #
#
#                             T R A I N                            #
#
# ------------------------------ ## ------------------------------ #
for it in range(train_state.step + 1, num_steps):
  key_train = keys_train[it % num_batches]
  x = env.reset(key_train)
  nx = env.step(x)
  (loss, metrics), grads = train_state.apply_fn(train_state.params, x, nx)
  # print(train_state.params)
  # print('____________________________________________________________________________________')
  train_state = train_state.apply_gradients(grads=grads)

  # metrics
  for key, value in metrics.items():
    sw.scalar(key, value, step=it)
  sw.flush()

  # eval
  if it % 100 == 0:
    plot_eval(train_state.params, km, x, 200, env.step)

sw.close()
# ------------------------------ ## ------------------------------ #


# In[ ]:


x = jnp.concatenate([env.reset(k) for k in jax.random.split(key_train, 1)], axis=0)
  nx = env.step(x)


# In[ ]:


plot_eval(train_state.params, km, x, 50, env.step)


# In[ ]:


x = jnp.arange(16, dtype=jnp.float32).reshape([4, 4])
eigvals_fn(x)


# In[ ]:




