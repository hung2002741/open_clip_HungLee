"""
CLIP model with LLM-based text encoder
This replaces the standard CLIP text encoder with a pretrained LLM (e.g., GPT-2, Phi, Gemma)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from collections import OrderedDict


class LLMTextEncoder(nn.Module):
    """
    Wrapper for using a pretrained LLM as the text encoder in CLIP.
    Supports models like GPT-2, Phi-2, Gemma, etc.
    """
    def __init__(
        self,
        model_name="gpt2",  # Can be: "gpt2", "microsoft/phi-2", "google/gemma-2b", etc.
        embed_dim=512,
        max_length=77,
        pooling_type="mean",  # "mean", "last", "cls", "attention"
        freeze_backbone=False,
        num_unfrozen_layers=4,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.pooling_type = pooling_type
        
        # Load pretrained LLM
        print(f"Loading LLM text encoder: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.llm = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True
        )
        
        # Get hidden dimension of the LLM
        self.hidden_dim = self.config.hidden_size
        
        # Projection layer to match CLIP's embedding dimension
        self.projection = nn.Linear(self.hidden_dim, embed_dim)
        
        # Optional: Layer norm before projection
        self.ln_pre = nn.LayerNorm(self.hidden_dim)
        
        # Freeze/unfreeze layers
        if freeze_backbone:
            self._freeze_layers(num_unfrozen_layers)
        
        # For attention pooling
        if pooling_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Softmax(dim=1)
            )
    
    def _freeze_layers(self, num_unfrozen_layers):
        """Freeze all layers except the last num_unfrozen_layers"""
        # Freeze all parameters first
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        if hasattr(self.llm, 'transformer'):  # GPT-2 style
            layers = self.llm.transformer.h
        elif hasattr(self.llm, 'model'):  # Some models wrap in .model
            if hasattr(self.llm.model, 'layers'):
                layers = self.llm.model.layers
            else:
                layers = self.llm.model.transformer.h
        elif hasattr(self.llm, 'layers'):
            layers = self.llm.layers
        else:
            print("Warning: Could not find layers to unfreeze. All parameters frozen.")
            return
        
        # Unfreeze last N layers
        for layer in layers[-num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"Frozen backbone, unfroze last {num_unfrozen_layers} layers")
    
    def pool_embeddings(self, hidden_states, attention_mask):
        """Pool token embeddings into a single vector"""
        if self.pooling_type == "last":
            # Use last token (like GPT style)
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, sequence_lengths]
            
        elif self.pooling_type == "cls":
            # Use first token (like BERT style)
            pooled = hidden_states[:, 0]
            
        elif self.pooling_type == "attention":
            # Learned attention pooling
            attention_weights = self.attention_pool(hidden_states)
            pooled = (hidden_states * attention_weights).sum(dim=1)
            
        else:  # "mean" or default
            # Mean pooling over non-padding tokens
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = (hidden_states * attention_mask_expanded).sum(dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - tokenized text
            attention_mask: [batch_size, seq_len] - attention mask
        
        Returns:
            text_features: [batch_size, embed_dim] - projected text embeddings
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Forward through LLM
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Pool embeddings
        pooled = self.pool_embeddings(hidden_states, attention_mask)
        
        # Layer norm and projection
        pooled = self.ln_pre(pooled)
        text_features = self.projection(pooled)
        
        # L2 normalize (important for CLIP)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features


class CLIPWithLLM(nn.Module):
    """
    CLIP model with LLM text encoder.
    Keeps the original vision encoder, replaces text encoder with LLM.
    """
    def __init__(
        self,
        vision_encoder,
        llm_model_name="gpt2",
        embed_dim=512,
        llm_max_length=77,
        llm_pooling_type="mean",
        freeze_llm_backbone=True,
        llm_unfrozen_layers=4,
        logit_scale_init=4.6052,  # ln(100), standard CLIP initialization
    ):
        super().__init__()
        
        # Vision encoder (from original CLIP)
        self.visual = vision_encoder
        
        # LLM text encoder (replacement)
        self.text_encoder = LLMTextEncoder(
            model_name=llm_model_name,
            embed_dim=embed_dim,
            max_length=llm_max_length,
            pooling_type=llm_pooling_type,
            freeze_backbone=freeze_llm_backbone,
            num_unfrozen_layers=llm_unfrozen_layers,
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        
    def encode_image(self, image):
        """Encode images"""
        return self.visual(image)
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text using LLM"""
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(self, image, input_ids, attention_mask=None):
        """
        Forward pass for contrastive learning
        
        Args:
            image: [batch_size, 3, H, W] - image tensor
            input_ids: [batch_size, seq_len] - tokenized text
            attention_mask: [batch_size, seq_len] - attention mask
        
        Returns:
            image_features: [batch_size, embed_dim]
            text_features: [batch_size, embed_dim]
            logit_scale: scalar
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)
        
        return image_features, text_features, self.logit_scale.exp()


class LLMTokenizer:
    """
    Wrapper for HuggingFace tokenizer to work with CLIP training pipeline
    """
    def __init__(self, model_name="gpt2", max_length=77):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, texts):
        """
        Tokenize texts for LLM
        
        Args:
            texts: list of strings or single string
        
        Returns:
            input_ids: tensor of shape [batch_size, max_length] or [max_length]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return input_ids (and optionally attention_mask)
        # For compatibility with existing CLIP code, return just input_ids
        # Attention mask will be created in the model if needed
        return encoded['input_ids']


def create_clip_with_llm(
    vision_model,
    llm_name="gpt2",
    embed_dim=512,
    freeze_vision=True,
    freeze_llm_backbone=True,
    llm_unfrozen_layers=4,
    llm_pooling="mean",
):
    """
    Factory function to create CLIP with LLM text encoder
    
    Args:
        vision_model: Pre-loaded vision encoder (e.g., from MobileCLIP)
        llm_name: Name of HuggingFace LLM model
        embed_dim: Embedding dimension for contrastive learning
        freeze_vision: Whether to freeze vision encoder
        freeze_llm_backbone: Whether to freeze LLM layers (except last N)
        llm_unfrozen_layers: Number of LLM layers to keep trainable
        llm_pooling: Pooling strategy ("mean", "last", "cls", "attention")
    
    Returns:
        model: CLIPWithLLM model
        tokenizer: LLMTokenizer
    """
    
    # Freeze vision encoder if requested
    if freeze_vision:
        for param in vision_model.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    # Create CLIP model with LLM
    model = CLIPWithLLM(
        vision_encoder=vision_model,
        llm_model_name=llm_name,
        embed_dim=embed_dim,
        freeze_llm_backbone=freeze_llm_backbone,
        llm_unfrozen_layers=llm_unfrozen_layers,
        llm_pooling_type=llm_pooling,
    )
    
    # Create tokenizer
    tokenizer = LLMTokenizer(model_name=llm_name, max_length=77)
    
    return model, tokenizer


# Example usage and recommended LLMs:
"""
Small LLMs (good for experimentation):
- "gpt2" (124M params) - Fast, widely used
- "gpt2-medium" (355M params)
- "distilgpt2" (82M params) - Lighter version

Medium LLMs (better performance):
- "microsoft/phi-2" (2.7B params) - Excellent quality
- "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params)

Multilingual:
- "bigscience/bloom-560m" (560M params)
- "xlm-roberta-base" (270M params)

Usage example:
```python
from open_clip import create_model_and_transforms

# Load vision encoder
vision_model, _, preprocess = create_model_and_transforms(
    'MobileCLIP2-B',
    pretrained='dfndr2b'
)

# Create CLIP with LLM
model, tokenizer = create_clip_with_llm(
    vision_model=vision_model.visual,
    llm_name="gpt2",
    embed_dim=512,
    freeze_vision=True,
    freeze_llm_backbone=True,
    llm_unfrozen_layers=4,
)

# Training
images = ...  # [B, 3, H, W]
texts = ["a photo of a cat", "a dog in the park"]
text_tokens = tokenizer(texts)

image_features, text_features, logit_scale = model(images, text_tokens)
```
"""