def save_model_bin(model_dir, bin_filepath):
    from bioamla.ast import load_pretrained_ast_model
    import torch
    model = load_pretrained_ast_model(model_dir)
    torch.save(model.state_dict(), bin_filepath)
    model.save_pretrained(model_dir)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    bin_filepath = sys.argv[2]
    save_model_bin(model_path, bin_filepath)