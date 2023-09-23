import argparse
import os

import sentence_transformers
import torch


def resize(
    sbert_input_uri: str,
    sbert_output_uri: str,
    new_num_tokens: int,
    bert_submodule_name: str,
    disable_tests: bool,
    dry_run: bool,
    extra_positions: int,
) -> None:
    sbert = sentence_transformers.SentenceTransformer(sbert_input_uri, device="cpu")
    bert = sbert.get_submodule(bert_submodule_name)

    old_pos_embs = bert.embeddings.position_embeddings
    new_pos_embs = bert._get_resized_embeddings(old_embeddings=old_pos_embs, new_num_tokens=new_num_tokens + extra_positions)
    bert.embeddings.position_embeddings = new_pos_embs
    bert.embeddings.position_ids = torch.arange(new_num_tokens + extra_positions).unsqueeze(0)

    old_token_ids = bert.embeddings.token_type_ids
    new_token_ids = torch.zeros(1, new_num_tokens, dtype=old_token_ids.dtype)
    bert.embeddings.token_type_ids = new_token_ids

    bert.config.max_position_embeddings = new_num_tokens + extra_positions
    bert.tie_weights()

    sbert.max_seq_length = new_num_tokens
    sbert.tokenizer.model_max_length = new_num_tokens

    sbert_output_uri = sbert_output_uri.rstrip("/")
    out_path, out_name = os.path.split(sbert_output_uri)

    if not dry_run:
        sbert.save(path=sbert_output_uri, model_name=out_name)
        print(f"Saved model at '{sbert_output_uri}'.")

    if not disable_tests:
        if not dry_run:
            sbert = sentence_transformers.SentenceTransformer(sbert_output_uri, device="cpu")

        test_embs = sbert.encode("Test small sentence", output_value="token_embeddings")
        assert test_embs.shape[0] < new_num_tokens, (test_embs.shape, test_embs.size)

        test_embs = sbert.encode(2 * new_num_tokens * "Test large sentence", output_value="token_embeddings")
        assert test_embs.shape[0] == new_num_tokens, (test_embs.shape, test_embs.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sbert_input_uri", type=str)
    parser.add_argument("sbert_output_uri", type=str)
    parser.add_argument("new_num_tokens", type=int)
    parser.add_argument("--bert-submodule-name", default="0.auto_model", type=str)
    parser.add_argument("--disable-tests", action="store_true")
    parser.add_argument("--extra-positions", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resize(**vars(args))
