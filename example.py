"""
$ python example.py
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 6.28MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 548M/548M [00:04<00:00, 115MB/s]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 1.37MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 338kB/s]
vocab.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:34<00:00, 30.3kB/s]
merges.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:14<00:00, 31.1kB/s]
tokenizer.json:   0%|                                                                                                                     | 0.00/1.36M [00:00<?, ?B/s]^[[B
Error while downloading from https://huggingface.co/gpt2/resolve/main/tokenizer.json: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out.
Trying to resume download...
tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.36M/1.36M [01:05<00:00, 20.7kB/s]
tokenizer.json:   0%|                                                                                                                     | 0.00/1.36M [02:57<?, ?B/s]
Loaded pretrained model gpt2 into HookedTransformer
resid_post_mlp tensor([6.4424e-05, 3.9219e-02, 3.1569e-02, 4.6318e-02, 7.1058e-02, 4.7744e-02,
        6.2675e-02, 6.7039e-02, 7.5507e-02], device='cuda:0')
"""
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(location, normalized_mse)

