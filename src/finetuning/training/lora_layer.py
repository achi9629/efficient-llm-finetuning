import os
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModelForCausalLM

class LoRALinear(nn.Module):
    def __init__(self, 
                 original_linear: nn.Linear, 
                 r: int,
                 alpha: int,
                 dropout: float = 0.0
        ) -> None:
        super(LoRALinear, self).__init__()
        
        """
        Description:
            Initialize a LoRA (Low-Rank Adaptation) linear layer.
            This constructor sets up a low-rank adapter for an existing linear layer,
            enabling efficient fine-tuning by decomposing weight updates into two
            low-rank matrices (A and B).
        Args:
            original_linear (nn.Linear): The original linear layer to adapt.
            r (int): The rank of the low-rank decomposition.
            alpha (int): The scaling factor for the LoRA updates.
            dropout (float, optional): Dropout rate to apply. Defaults to 0.0.
        Returns:
            None
        Attributes:
            original_linear (nn.Linear): Reference to the original linear layer.
            r (int): Rank of the low-rank matrices.
            alpha (int): Scaling factor.
            scaling (float): Computed scaling value (alpha / r).
            lora_A (nn.Parameter): Low-rank matrix A with shape (r, in_features),
                                    initialized with Kaiming uniform distribution.
            lora_B (nn.Parameter): Low-rank matrix B with shape (out_features, r),
                                    initialized to zeros.
            dropout (nn.Dropout): Dropout layer for regularization.
        Note:
            The original linear layer's weights and biases are frozen
            (requires_grad set to False) to prevent gradient updates during training.
        """
        
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        self.dropout_val = dropout
        
        # Initialize the low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(self.r, self.original_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.original_linear.out_features, self.r))
        self.dropout = nn.Dropout(dropout)
        
        # Apply Kaiming uniform initialization
        init.kaiming_uniform_(self.lora_A, a=0, mode='fan_in', nonlinearity='relu')
        
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Description:
            Forward pass for LoRA (Low-Rank Adaptation) layer.
            Computes the output by combining the base linear layer output with the LoRA adaptation.
            The LoRA adaptation applies dropout to the input, then projects it through two low-rank
            matrices (A and B) and scales the result.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).
        Returns:
            torch.Tensor: Output tensor combining base linear output with LoRA adaptation,
                         same shape as input.
        """
        
        base_output = self.original_linear(x)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return base_output + lora_output
    
def inject_lora(model: AutoModelForCausalLM, 
                target_modules: list[str],
                r: int, 
                alpha: int, 
                dropout: float
    ) -> AutoModelForCausalLM:
    
    """
    Description:
        Injects LoRA (Low-Rank Adaptation) layers into a model's target modules.
        This function replaces specified Linear layers in the model with LoRA-adapted versions,
        enabling efficient fine-tuning by adding trainable low-rank decomposition matrices
        alongside frozen pre-trained weights.
    Args:
        model (AutoModelForCausalLM): The pre-trained causal language model to modify.
        target_modules (list[str]): List of module name patterns to target for LoRA injection.
                                    Modules matching these patterns will be wrapped with LoRA.
        r (int): The rank of the LoRA decomposition matrices. Lower values reduce parameters.
        alpha (float): The scaling factor for LoRA updates. Controls the magnitude of LoRA contributions.
        dropout (float): Dropout probability applied to LoRA layers for regularization.
    Returns:
        AutoModelForCausalLM: The modified model with LoRA layers injected into target modules.
    """
    
    # Collect targets first to avoid modifying during iteration
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            targets.append((name, module))
            
    for name, module in targets:
        
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            child_name = name
        lora_layer = LoRALinear(module, r, alpha, dropout)
        setattr(parent_module, child_name, lora_layer)
        
    return model

def merge_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    
    """
    Description:
        Merge LoRA weights into the base model weights and replace LoRA layers with standard linear layers.
        This function iterates through all LoRA layers in the model, computes the merged weights by adding
        the LoRA-adapted weights to the original weights, and replaces the LoRA layers with standard linear
        layers containing the merged weights.
    Args:
        model (AutoModelForCausalLM): The model containing LoRA layers to be merged.
    Returns:
        AutoModelForCausalLM: The model with LoRA layers replaced by merged linear layers.
    Note:
        - All weight operations are performed without gradient computation (no_grad context).
        - LoRA weights are computed as: scaling * (lora_B @ lora_A)
        - The resulting weights are converted back to the original dtype before assignment.
        - This operation is performed in-place but the model is also returned for convenience.
    """
    
    # Collect targets first to avoid modifying during iteration
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            targets.append((name, module))
    
    for name, module in targets:
            
        with torch.no_grad():
            
            original_weights = module.original_linear.weight.data.float()
            lora_weights = module.scaling * (module.lora_B.float() @ module.lora_A.float())
            new_weights = (original_weights + lora_weights).to(module.original_linear.weight.data.dtype)
            
            new_linear = module.original_linear
            new_linear.weight.data.copy_(new_weights)
            
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
                child_name = name
            setattr(parent_module, child_name, new_linear)
                
    return model

def save_lora_weights(model: AutoModelForCausalLM, 
                      save_path: str
    ) -> None:
    
    """
    Save LoRA weights from a model to a file.
    This function extracts all LoRA weights (lora_A and lora_B) from LoRALinear 
    modules in the model along with their configuration parameters (rank, alpha, 
    and dropout), and saves them to the specified path.
    Args:
        model (AutoModelForCausalLM): The model containing LoRA layers to save.
        save_path (str): The file path where the LoRA weights will be saved.
    Returns:
        None
    Raises:
        IOError: If the save_path is not writable or the directory does not exist.
    Note:
        The function assumes that the model contains at least one LoRALinear module.
        All tensors are moved to CPU and detached from the computation graph before saving.
    """
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except OSError as e:
        raise IOError(f"Failed to create directory for save path: {save_path}") from e
    
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Both clone and detach but we use clone to ensure we have a separate copy of 
            # the data that is not shared with the original tensor
            state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu().clone()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu().clone()
            if 'r' not in state_dict:
                state_dict['r'] = torch.tensor(module.r)
            if 'alpha' not in state_dict:
                state_dict['alpha'] = torch.tensor(module.alpha)
            if 'dropout' not in state_dict:
                state_dict['dropout'] = torch.tensor(module.dropout_val)
    
    torch.save(state_dict, save_path)
    
def load_lora_weights(model: AutoModelForCausalLM, 
                      path: str, 
                      target_modules: list[str]
    ) -> AutoModelForCausalLM:
    
    """
    Descriiption:
        Load LoRA weights from a saved checkpoint and inject them into a model.
        This function loads pre-trained LoRA (Low-Rank Adaptation) parameters from a file,
        injects LoRA layers into the specified target modules of the model, and restores
        the trained weights.
    Args:
        model (AutoModelForCausalLM): The base causal language model to load LoRA weights into.
        path (str): File path to the saved LoRA checkpoint containing weights and hyperparameters.
        target_modules (list[str]): List of module names where LoRA layers should be injected.
    Returns:
        AutoModelForCausalLM: The model with LoRA layers injected and weights loaded from checkpoint.
    Raises:
        FileNotFoundError: If the checkpoint file at `path` does not exist.
        KeyError: If required keys ('r', 'alpha', 'dropout') are missing from the checkpoint.
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>> model = load_lora_weights(model, "checkpoint.pt", ["q_proj", "v_proj"])
    """
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LoRA checkpoint file not found at: {path}")
    
    state_dict = torch.load(path, weights_only = True)
    
    if 'r' not in state_dict or 'alpha' not in state_dict or 'dropout' not in state_dict:
        raise KeyError("Checkpoint must contain 'r', 'alpha', and 'dropout' keys.")
    
    r = state_dict['r'].item() 
    alpha = state_dict['alpha'].item()
    dropout = state_dict['dropout'].item()
    
    model = inject_lora(model, target_modules, r, alpha, dropout)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.copy_(state_dict[f"{name}.lora_A"])
            module.lora_B.data.copy_(state_dict[f"{name}.lora_B"])
    
    return model