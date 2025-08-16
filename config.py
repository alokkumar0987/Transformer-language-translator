from pathlib import Path

def get_config():
    return {
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "seq_len": 64,
        "d_model": 512,
        "batch_size": 4,
        "num_epochs": 15,
        "lr": 1e-4,
        "model_folder": "weights",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer-en-it",
        "preload": None,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])