Optional: model folders for building the embedder image without access to the model hub.
Put each model in a folder named by its registry key (for example `all-mpnet-base-v2/` with `model.safetensors`,
`config.json`, tokenizer files and the sentence-transformers configuration). The build verifies `model.safetensors`
against the checksum in `docintel/model_registry.yaml`. Model files are not tracked by git.
