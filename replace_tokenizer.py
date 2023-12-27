import os
import glob
import argparse
import shutil
import json

import numpy as np
import transformers
import sentence_transformers


def run(target_model_uri: str, new_tokenizer_uri: str, dry_run: bool) -> None:
    if not os.path.isdir(new_tokenizer_uri):
        raise ValueError(f"{new_tokenizer_uri=} must be a directory.")
    if not os.path.isdir(target_model_uri):
        raise ValueError(f"{target_model_uri=} must be a directory.")

    config_json_uri = os.path.join(target_model_uri, "config.json")

    if not os.path.exists(config_json_uri):
        raise OSError(f"Unable to find '{config_json_uri=}'.")

    with open(os.path.join(new_tokenizer_uri, "vocab.txt"), "r", encoding="utf-8") as f_in:
        new_vocab_size = int(sum(1 for _ in f_in))

    print("New vocabulary size:", new_vocab_size)
    assert new_vocab_size >= 25000

    def fn(dry_run: bool, quiet: bool) -> None:
        for input_uri in glob.glob(os.path.join(new_tokenizer_uri, "*")):
            file_urn = os.path.basename(input_uri)
            output_uri = os.path.join(target_model_uri, file_urn)
            if not os.path.exists(output_uri):
                raise OSError(f"File {output_uri=} doesn't exist in {target_model_uri=}.")

            if not dry_run:
                shutil.copyfile(src=input_uri, dst=output_uri)
                if not quiet:
                    print(f"Replaced '{output_uri}'.")
            else:
                if not quiet:
                    print(f"Would replace '{output_uri}' (dry run is activated).")

    fn(dry_run=True, quiet=not dry_run)

    if not dry_run:
        sbert = sentence_transformers.SentenceTransformer(target_model_uri, device="cpu")
        embs_orig = sbert._first_module().auto_model.embeddings.word_embeddings
        orig_vocab_size, emb_dim = (embs_orig.num_embeddings, embs_orig.embedding_dim)
        num_param_orig = sum(item.numel() for item in sbert.parameters())
        embs_orig = sbert.encode("Olá")

        config = transformers.BertConfig.from_pretrained(target_model_uri)
        new_bert = transformers.BertModel(config)
        new_bert.resize_token_embeddings(new_vocab_size)
        new_bert.save_pretrained(target_model_uri)
        fn(dry_run=False, quiet=False)

        sbert_config_uri = os.path.join(target_model_uri, "sentence_bert_config.json")
        with open(sbert_config_uri, "r", encoding="utf-8") as f_in:
            sbert_config = json.load(f_in)

        sbert_config["max_seq_length"] = 512
        sbert_config["do_lower_case"] = False

        with open(sbert_config_uri, "w") as f_out:
            json.dump(sbert_config, f_out, indent=2)

        sbert = sentence_transformers.SentenceTransformer(target_model_uri, device="cpu")
        num_param_new = sum(item.numel() for item in sbert.parameters())
        embs_new = sbert.encode("Olá")

        assert sbert.tokenizer.vocab_size == new_vocab_size
        assert num_param_orig == num_param_new + ((orig_vocab_size - new_vocab_size) * emb_dim)
        assert not np.allclose(embs_orig, embs_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_model_uri", type=str)
    parser.add_argument("new_tokenizer_uri", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(**vars(args))
