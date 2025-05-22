from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class TrainingStrategy(str, Enum):
    """Available training strategies."""
    FULL_FINE_TUNING = "full_fine_tuning"
    PEFT = "peft"  # Parameter-Efficient Fine-Tuning
    LORA = "lora"  # Low-Rank Adaptation
    QLORA = "qlora"  # Quantized LoRA
    CONTINUAL = "continual"  # Continual Learning

class StrategyConfig(BaseModel):
    """Base configuration for training strategies."""
    strategy: TrainingStrategy
    learning_rate: float = Field(default=1e-5)
    num_epochs: int = Field(default=3)
    batch_size: int = Field(default=8)
    gradient_accumulation_steps: int = Field(default=4)
    warmup_steps: int = Field(default=100)
    weight_decay: float = Field(default=0.01)
    max_grad_norm: float = Field(default=1.0)

class FullFineTuningConfig(StrategyConfig):
    """Configuration for full fine-tuning."""
    strategy: TrainingStrategy = TrainingStrategy.FULL_FINE_TUNING
    unfreeze_layers: Optional[int] = Field(default=None)  # Number of layers to unfreeze from end

class PEFTConfig(StrategyConfig):
    """Configuration for Parameter-Efficient Fine-Tuning."""
    strategy: TrainingStrategy = TrainingStrategy.PEFT
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    r: int = Field(default=8)  # Rank of update matrices
    alpha: int = Field(default=16)  # Scaling factor
    dropout: float = Field(default=0.1)

class LoRAConfig(PEFTConfig):
    """Configuration for Low-Rank Adaptation."""
    strategy: TrainingStrategy = TrainingStrategy.LORA
    use_rslora: bool = Field(default=False)  # Use Rank-Stabilized LoRA
    use_dora: bool = Field(default=False)  # Use Dynamic Rank Adaptation

class QLoRAConfig(LoRAConfig):
    """Configuration for Quantized LoRA."""
    strategy: TrainingStrategy = TrainingStrategy.QLORA
    bits: int = Field(default=4)  # Number of bits for quantization
    double_quant: bool = Field(default=True)
    quant_type: str = Field(default="nf4")  # Normal Float 4

class ContinualConfig(StrategyConfig):
    """Configuration for Continual Learning."""
    strategy: TrainingStrategy = TrainingStrategy.CONTINUAL
    memory_size: int = Field(default=1000)  # Size of memory buffer
    replay_strategy: str = Field(default="random")  # Strategy for replay
    importance_sampling: bool = Field(default=True)

def get_strategy_config(strategy: TrainingStrategy, **kwargs) -> StrategyConfig:
    """
    Get the appropriate configuration for a training strategy.
    
    Args:
        strategy: Training strategy to use
        **kwargs: Additional configuration parameters
        
    Returns:
        StrategyConfig instance
    """
    strategy_map = {
        TrainingStrategy.FULL_FINE_TUNING: FullFineTuningConfig,
        TrainingStrategy.PEFT: PEFTConfig,
        TrainingStrategy.LORA: LoRAConfig,
        TrainingStrategy.QLORA: QLoRAConfig,
        TrainingStrategy.CONTINUAL: ContinualConfig
    }
    
    config_class = strategy_map.get(strategy)
    if not config_class:
        raise ValueError(f"Unknown training strategy: {strategy}")
    
    return config_class(**kwargs)

def get_hyperparameters(config: StrategyConfig) -> Dict[str, Any]:
    """
    Convert strategy configuration to hyperparameters for training.
    
    Args:
        config: Strategy configuration
        
    Returns:
        Dict of hyperparameters
    """
    base_params = {
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm
    }
    
    strategy_params = {}
    if isinstance(config, FullFineTuningConfig):
        strategy_params = {
            "unfreeze_layers": config.unfreeze_layers
        }
    elif isinstance(config, PEFTConfig):
        strategy_params = {
            "target_modules": config.target_modules,
            "r": config.r,
            "alpha": config.alpha,
            "dropout": config.dropout
        }
    elif isinstance(config, LoRAConfig):
        strategy_params = {
            "target_modules": config.target_modules,
            "r": config.r,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "use_rslora": config.use_rslora,
            "use_dora": config.use_dora
        }
    elif isinstance(config, QLoRAConfig):
        strategy_params = {
            "target_modules": config.target_modules,
            "r": config.r,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "use_rslora": config.use_rslora,
            "use_dora": config.use_dora,
            "bits": config.bits,
            "double_quant": config.double_quant,
            "quant_type": config.quant_type
        }
    elif isinstance(config, ContinualConfig):
        strategy_params = {
            "memory_size": config.memory_size,
            "replay_strategy": config.replay_strategy,
            "importance_sampling": config.importance_sampling
        }
    
    return {**base_params, **strategy_params} 