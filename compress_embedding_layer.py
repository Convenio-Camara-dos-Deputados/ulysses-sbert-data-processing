# TODO: refactor code
import os
import json
import collections
import shutil
import copy

import sentence_transformers
import torch.nn


model_uri = os.path.expanduser("~/hdd/compare_sentence_models/sentence-transformers_LaBSE")
model_uri = os.path.abspath(model_uri)
model = sentence_transformers.SentenceTransformer(model_uri, device="cpu")
model.eval()


with open("used_tokens_dist.json", "r") as f_in:
    used_tokens_in_ulysses_subset = json.load(f_in, object_hook=collections.Counter)


output_dir = os.path.abspath("./compressed_models")
os.makedirs(output_dir, exist_ok=True)

special_tokens = [k for k, v in model.tokenizer.get_vocab().items() if k.startswith("[") and k.endswith("]")]
special_tokens_token_to_id = {special_token: model.tokenizer.vocab[special_token] for special_token in special_tokens}
special_tokens_id_to_token = {v: k for k, v in special_tokens_token_to_id.items()}


with torch.no_grad():
    word_embeddings = model._first_module().get_submodule("auto_model.embeddings.word_embeddings")
    word_embeddings.requires_grad_(False)
    print(word_embeddings.padding_idx)

    for new_dict_size in (5000, 10000, 15000, 30000):
        model_name = os.path.basename(model_uri) + "_" + str(new_dict_size)
        final_output_dir = os.path.join(output_dir, model_name)

        if os.path.exists(final_output_dir):
            print(f"Skipping cached {final_output_dir}.")
            continue

        most_common_token_ids = sorted([int(k) for k, _ in used_tokens_in_ulysses_subset.most_common(new_dict_size)])

        new_emb = torch.nn.modules.sparse.Embedding(
            num_embeddings=new_dict_size + len(special_tokens),
            embedding_dim=word_embeddings.embedding_dim,
            padding_idx=word_embeddings.padding_idx,
        )
        new_emb.requires_grad_(False)
        new_emb.weight *= 0.0

        new_vocab = {}

        for special_token_id in special_tokens_id_to_token:
            # Keep special tokens in the same original index.
            new_emb.weight[special_token_id, :] = word_embeddings.weight[special_token_id, :]
            new_vocab[special_tokens_id_to_token[special_token_id]] = special_token_id

        i = 0
        while i in special_tokens_id_to_token:
            print(f"Skipped token_id={i}.")
            i += 1

        for token_id in most_common_token_ids:
            if token_id in special_tokens_id_to_token:
                continue

            new_emb.weight[i, :] = word_embeddings.weight[token_id, :]
            new_vocab[model.tokenizer.decode(token_id)] = i

            i += 1
            while i in special_tokens_id_to_token:
                print(f"Skipped token_id={i}.")
                i += 1

        new_model = copy.deepcopy(model)

        with open("new_vocab.temp", "w") as f_out:
            f_out.write("\n".join([k for k, _ in sorted(new_vocab.items(), key=lambda item: int(item[1]))]))

        new_model.tokenizer._tokenizer.model = new_model.tokenizer._tokenizer.model.from_file("new_vocab.temp")

        new_bert = new_model._first_module()
        new_bert.auto_model.resize_token_embeddings(new_emb.num_embeddings)
        new_bert.auto_model.embeddings.word_embeddings = new_emb

        new_model.save(path=final_output_dir, model_name=model_name)

        sentence_transformers.SentenceTransformer(final_output_dir, device="cpu")
