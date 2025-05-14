
def ablate_attention_head(output, head_idx, num_heads):
    """
    Ablates a specific attention head by zeroing its contribution in the output.
    """
    # Reshape the output to separate heads (assuming [batch_size, seq_len, embed_dim])
    # Ouput is a tuple containing attn_output, attn_output_weights. We only care about the attn_output
    batch_size, seq_len, embed_dim = output[0].shape
    head_dim = embed_dim // num_heads
    reshaped_output = output[0].view(batch_size, seq_len, num_heads, head_dim)
    
    # Zero out the specified head
    reshaped_output[:, :, head_idx, :] = 0
    
    # Reshape back to original dimensions & return with (original) attention weights
    return reshaped_output.view(batch_size, seq_len, embed_dim), output[1]
