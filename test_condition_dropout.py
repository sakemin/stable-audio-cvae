#!/usr/bin/env python3
"""
Test script to demonstrate condition dropout functionality.
"""

import torch
from stable_audio_tools.models.cvae import ConditionEmbedding, AudioCVAE

def test_condition_embedding_dropout():
    """Test the ConditionEmbedding with dropout."""
    print("🧪 Testing ConditionEmbedding with dropout...")
    
    # Setup
    n_classes = [5]  # Single categorical condition with 5 classes (0-4)
    embed_dim = 128
    dropout_prob = 0.5  # High probability for testing
    batch_size = 8
    
    # Create embedding with dropout
    cond_embed = ConditionEmbedding(n_classes, embed_dim, dropout_prob)
    
    # Create condition tensor (class indices 0-4)
    conditions = torch.randint(0, 5, (batch_size, 1))
    print(f"Original conditions: {conditions.flatten().tolist()}")
    
    # Test training mode (with dropout)
    cond_embed.train()
    embedding_train = cond_embed(conditions, training=True)
    print(f"Training mode embedding shape: {embedding_train.shape}")
    
    # Test eval mode (no dropout)
    cond_embed.eval()
    embedding_eval = cond_embed(conditions, training=False)
    print(f"Eval mode embedding shape: {embedding_eval.shape}")
    
    # Test that null condition (index 5) works
    null_conditions = torch.full((batch_size, 1), 5)  # null token
    embedding_null = cond_embed(null_conditions, training=False)
    print(f"Null condition embedding shape: {embedding_null.shape}")
    
    print("✅ ConditionEmbedding dropout test passed!\n")


def test_audio_cvae_dropout():
    """Test the full AudioCVAE with condition dropout."""
    print("🧪 Testing AudioCVAE with condition dropout...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    n_channels = 2
    n_samples = 65536
    n_classes = 5
    
    # Create model with dropout
    model = AudioCVAE(
        cond_def=[n_classes],
        cond_embed_dim=128,
        cond_dropout_prob=0.3,
        encoder_cfg={'in_channels': n_channels, 'channels': 128, 'c_mults': [1, 2, 4, 8, 16], 
                    'strides': [2, 4, 4, 8, 8], 'latent_dim': 128, 'use_snake': True},
        decoder_cfg={'out_channels': n_channels, 'channels': 128, 'c_mults': [1, 2, 4, 8, 16],
                    'strides': [2, 4, 4, 8, 8], 'latent_dim': 64, 'use_snake': True, 'final_tanh': False}
    ).to(device)
    
    # Create dummy data
    audio = torch.randn(batch_size, n_channels, n_samples, device=device)
    conditions = torch.randint(0, n_classes, (batch_size, 1), device=device)
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Input conditions: {conditions.flatten().tolist()}")
    
    # Test training mode (with condition dropout)
    model.train()
    recon_train, kl_train = model(audio, conditions)
    print(f"Training reconstruction shape: {recon_train.shape}")
    print(f"Training KL divergence: {kl_train.item():.4f}")
    
    # Test eval mode (no condition dropout)
    model.eval()
    with torch.no_grad():
        recon_eval, kl_eval = model(audio, conditions)
        print(f"Eval reconstruction shape: {recon_eval.shape}")
        print(f"Eval KL divergence: {kl_eval.item():.4f}")
    
    # Test unconditional generation
    print("\n🎵 Testing unconditional generation...")
    unconditional_samples = model.generate_unconditional(batch_size=2, device=device)
    print(f"Unconditional samples shape: {unconditional_samples.shape}")
    
    print("✅ AudioCVAE dropout test passed!\n")


if __name__ == "__main__":
    print("🚀 Starting condition dropout tests...\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    test_condition_embedding_dropout()
    test_audio_cvae_dropout()
    
    print("🎉 All tests completed successfully!")
    print("\n💡 Key features implemented:")
    print("  • Condition dropout during training (randomly replaces conditions with null token)")
    print("  • Null condition token (index 5 for the 5-class categorical condition)")
    print("  • Unconditional generation capability")
    print("  • Configurable dropout probability via --cond_dropout_prob argument") 