import torch
import pytest, logging
import torch.nn as nn

from src.finetuning import LoRALinear, inject_lora, merge_lora, save_lora_weights, load_lora_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

in_features = 64
out_features = 128

@pytest.fixture
def tinymodel():
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleDict({
                'self_attn': nn.ModuleDict({
                    'q_proj': nn.Linear(in_features, out_features, bias = False),
                    'k_proj': nn.Linear(in_features, out_features, bias = False),
                    'v_proj': nn.Linear(in_features, out_features, bias = False),
                    'o_proj': nn.Linear(in_features, out_features, bias = False),
                }),
                'mlp': nn.ModuleDict({
                    'gate_proj': nn.Linear(64, 128),
                    'up_proj': nn.Linear(64, 128),
                })
            })
            
        def forward(self, x):
            q = self.layer['self_attn']['q_proj'](x)
            k = self.layer['self_attn']['k_proj'](x)
            v = self.layer['self_attn']['v_proj'](x)
            o = self.layer['self_attn']['o_proj'](x)
            gate = self.layer['mlp']['gate_proj'](x)
            up = self.layer['mlp']['up_proj'](x)
            return q + k + v + o + gate + up
        
    return TinyModel

def test_loralinear_shape():
    
    in_features = 512
    out_features = 256
    r = 4
    alpha = 16
    x = torch.randn(8, in_features)
    
    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(original_linear = linear, 
                             r = r, 
                             alpha=alpha
                    )
    lora_linear.eval()  # disable dropout so test is deterministic
    
    with torch.inference_mode():
        linear_out = linear(x)
        lora_out = lora_linear(x)
    
    
    assert linear_out.shape == lora_out.shape, f"Expected output shape {linear_out.shape}, but got {lora_out.shape}"
    
def test_loralinear_zero_init():
    
    in_features = 512
    out_features = 256
    r = 4
    alpha = 16
    x = torch.randn(8, in_features)
    
    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(original_linear = linear, 
                             r = r, 
                             alpha=alpha
                    )
    lora_linear.eval()  # disable dropout so test is deterministic

    with torch.inference_mode():
        linear_out = linear(x)
        lora_out = lora_linear(x)
    
    assert torch.allclose(linear_out, lora_out, atol=1e-6), "Expected LoRA output to match original linear output when LoRA weights are zero-initialized"
    
def test_loralinear_freezes_original():
    
    in_features = 512
    out_features = 256
    r = 4
    alpha = 16
    
    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(original_linear = linear, 
                             r = r, 
                             alpha=alpha
                    )
    lora_linear.eval()  # disable dropout so test is deterministic

    assert lora_linear.original_linear.weight.requires_grad == False, "Expected original linear weights to be frozen (requires_grad=False)"
    if lora_linear.original_linear.bias is not None:
        assert lora_linear.original_linear.bias.requires_grad == False, "Expected original linear bias to be frozen (requires_grad=False)"
        
def test_loralinear_lora_params_trainable():
    
    in_features = 512
    out_features = 256
    r = 4
    alpha = 16
    
    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(original_linear = linear, 
                             r = r, 
                             alpha=alpha
                    )
    lora_linear.eval()  # disable dropout so test is deterministic

    assert lora_linear.lora_A.requires_grad == True, "Expected LoRA A weights to be trainable (requires_grad=True)"
    assert lora_linear.lora_B.requires_grad == True, "Expected LoRA B weights to be trainable (requires_grad=True)"
    
def test_loralinear_scaling():
    
    in_features = 512
    out_features = 256
    r = 4
    alpha = 16
    
    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(original_linear = linear, 
                             r = r, 
                             alpha=alpha
                    )
    lora_linear.eval()  # disable dropout so test is deterministic

    assert lora_linear.scaling == alpha / r, f"Expected scaling factor to be {alpha / r}, but got {lora_linear.scaling}"

def test_inject_lora_replaces_targets(tinymodel):

    model = tinymodel()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = 4, 
                        alpha = 16, 
                        dropout = 0.0
            )
    
    # Targeted modules should be LoRALinear
    assert isinstance(model.layer['self_attn']['q_proj'], LoRALinear), "Expected q_proj to be replaced with LoRALinear"
    assert isinstance(model.layer['self_attn']['k_proj'], LoRALinear), "Expected k_proj to be replaced with LoRALinear"
    assert isinstance(model.layer['self_attn']['v_proj'], LoRALinear), "Expected v_proj to be replaced with LoRALinear"
    assert isinstance(model.layer['self_attn']['o_proj'], LoRALinear), "Expected o_proj to be replaced with LoRALinear"
    
    # Non-targeted modules should remain nn.Linear
    assert isinstance(model.layer['mlp']['gate_proj'], nn.Linear), "Expected gate_proj to remain nn.Linear"
    assert isinstance(model.layer['mlp']['up_proj'], nn.Linear), "Expected up_proj to remain nn.Linear"
    
def test_inject_lora_param_count(tinymodel):
    
    model = tinymodel()
    r = 4
    alpha = 16
    dropout = 0.0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = dropout
            )

    # Total trainable parameters should be close to (in_features * r + out_features * r) * number of target modules
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params_per_module = (in_features * r + out_features * r)
    expected_total_params = expected_params_per_module * len(target_modules)
    
    assert trainable_params == expected_total_params, f"Expected {expected_total_params} trainable parameters, but got {trainable_params}"
    
def test_merge_produces_linear(tinymodel):
    
    model = tinymodel()
    r = 4
    alpha = 16
    dropout = 0.0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = dropout
            )
    
    # targeted modules should be LoRALinear before merging
    assert isinstance(model.layer['self_attn']['q_proj'], LoRALinear), "Expected q_proj to be LoRALinear before merging"
    assert isinstance(model.layer['self_attn']['k_proj'], LoRALinear), "Expected k_proj to be LoRALinear before merging"
    assert isinstance(model.layer['self_attn']['v_proj'], LoRALinear), "Expected v_proj to be LoRALinear before merging"
    assert isinstance(model.layer['self_attn']['o_proj'], LoRALinear), "Expected o_proj to be LoRALinear before merging"
    
    model = merge_lora(model)
    
    # Targeted modules should be nn.Linear after merging
    assert isinstance(model.layer['self_attn']['q_proj'], nn.Linear), "Expected q_proj to be nn.Linear after merging"
    assert isinstance(model.layer['self_attn']['k_proj'], nn.Linear), "Expected k_proj to be nn.Linear after merging"
    assert isinstance(model.layer['self_attn']['v_proj'], nn.Linear), "Expected v_proj to be nn.Linear after merging"
    assert isinstance(model.layer['self_attn']['o_proj'], nn.Linear), "Expected o_proj to be nn.Linear after merging"
    
def test_merge_correctness(tinymodel):
    
    model = tinymodel()
    r = 4
    alpha = 16
    dropout = 0.0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    x = torch.randn(8, in_features)
    
    # Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = dropout
            )
    
    with torch.inference_mode():
        out_after_inject = model(x)
        
    model = merge_lora(model)
    
    with torch.inference_mode():
        out_after_merge = model(x)
        
    assert torch.allclose(out_after_inject, out_after_merge, atol=1e-6), "Expected outputs to match after merging LoRA weights into original linear layers"
    
def test_save_load_roundtrip(tinymodel, tmp_path):
    
    model = tinymodel()
    r = 4
    alpha = 16
    dropout = 0.0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    x = torch.randn(8, in_features)
    
    save_full_model_path = tmp_path / "full_model.pt"
    torch.save(model.state_dict(), save_full_model_path)  # Save full model state dict for debugging
    
    model = inject_lora(model = model, 
                        target_modules = target_modules, 
                        r = r, 
                        alpha = alpha, 
                        dropout = dropout
            )
    
    model.eval()
    
    with torch.inference_mode():
        out_before = model(x)
        
    # Save
    save_path = tmp_path / "lora_weights.pt"
    save_lora_weights(model, str(save_path))
    
    # Create new model and load weights
    new_model = tinymodel()
    model_weights = torch.load(save_full_model_path, weights_only = True)  # Load full model weights for debugging
    new_model.load_state_dict(model_weights)  # Load full model weights to ensure architecture matches
    new_model = load_lora_weights(new_model, save_path, target_modules)
    new_model.eval()
    
    with torch.inference_mode():
        out_after = new_model(x)
    
    assert torch.allclose(out_before, out_after, atol=1e-8), "Expected outputs to match before saving and after loading LoRA weights"