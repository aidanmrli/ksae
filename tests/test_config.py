"""Tests for the config system (config.py).

Tests configuration loading, modification, and registry functionality.
"""

import pytest

from config import (
    get_config,
    get_default_config,
    get_train_generic_km_config,
    get_train_lista_config,
)


def test_get_default_config():
    """Test that default config has expected structure and values."""
    cfg = get_default_config()
    
    # Check top-level keys
    assert hasattr(cfg, "SEED")
    assert hasattr(cfg, "ENV")
    assert hasattr(cfg, "MODEL")
    assert hasattr(cfg, "TRAIN")
    
    # Check ENV structure
    assert hasattr(cfg.ENV, "ENV_NAME")
    assert hasattr(cfg.ENV, "DUFFING")
    assert hasattr(cfg.ENV, "PARABOLIC")
    
    # Check MODEL structure
    assert hasattr(cfg.MODEL, "MODEL_NAME")
    assert hasattr(cfg.MODEL, "TARGET_SIZE")
    assert hasattr(cfg.MODEL, "ENCODER")
    assert hasattr(cfg.MODEL, "DECODER")
    
    # Check loss coefficients
    assert hasattr(cfg.MODEL, "RES_COEFF")
    assert hasattr(cfg.MODEL, "RECONST_COEFF")
    assert hasattr(cfg.MODEL, "PRED_COEFF")
    assert hasattr(cfg.MODEL, "SPARSITY_COEFF")
    
    # Check TRAIN structure
    assert hasattr(cfg.TRAIN, "NUM_STEPS")
    assert hasattr(cfg.TRAIN, "BATCH_SIZE")
    assert hasattr(cfg.TRAIN, "LR")


def test_get_named_configs():
    """Test that named configurations load correctly."""
    # Test generic config
    cfg_generic = get_train_generic_km_config()
    assert cfg_generic.MODEL.MODEL_NAME == "GenericKM"
    assert cfg_generic.MODEL.TARGET_SIZE == 64
    
    # Test LISTA config
    cfg_lista = get_train_lista_config()
    assert cfg_lista.MODEL.MODEL_NAME == "KSAE"
    assert cfg_lista.MODEL.ENCODER.LISTA.NUM_LOOPS == 10
    assert cfg_lista.MODEL.TARGET_SIZE == 1024 * 2


def test_config_registry():
    """Test that config registry works."""
    cfg = get_config("default")
    assert cfg is not None
    
    cfg_generic = get_config("generic")
    assert cfg_generic.MODEL.MODEL_NAME == "GenericKM"
    
    cfg_lista = get_config("lista")
    assert cfg_lista.MODEL.MODEL_NAME == "KSAE"
    
    with pytest.raises(ValueError):
        get_config("nonexistent")


def test_config_modification():
    """Test that config can be modified."""
    cfg = get_default_config()
    original_lr = cfg.TRAIN.LR
    
    cfg.TRAIN.LR = 1e-4
    assert cfg.TRAIN.LR == 1e-4
    assert cfg.TRAIN.LR != original_lr


def test_lista_config():
    """Test LISTA-specific configuration parameters."""
    cfg = get_train_lista_config()
    
    # Check LISTA-specific fields exist and have correct values
    assert cfg.MODEL.ENCODER.LISTA.NUM_LOOPS == 10
    assert cfg.MODEL.ENCODER.LISTA.L == 1e4
    assert cfg.MODEL.ENCODER.LISTA.ALPHA == 0.1
    assert cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER is True


def test_config_dt_extraction():
    """Test that dt is correctly set in environment-specific configs."""
    cfg = get_default_config()
    cfg.ENV.ENV_NAME = "duffing"
    cfg.ENV.DUFFING.DT = 0.02
    
    # Check that dt is correctly set
    assert cfg.ENV.DUFFING.DT == 0.02
    assert cfg.ENV.ENV_NAME == "duffing"

