import collections
import argparse
import hashlib
import glob
import json
import copy
import os

import sentence_transformers
import torch.nn
import tqdm


def build_cache_uri(model_uri: str, corpora_uris: list[str], cache_dir: str) -> str:
    corpora_urns = [os.path.basename(item.rstrip("/")) for item in corpora_uris]
    corpora_urns.sort()

    hasher = hashlib.sha256()
    for urn in corpora_urns:
        hasher.update(urn.encode())

    corpora_urns_sha256 = hasher.hexdigest()
    model_urn = os.path.basename(model_uri.rstrip("/"))
    return os.path.join(cache_dir, f"{model_urn}_{corpora_urns_sha256}.npy")


def compute_token_id_dist(
    corpora_uris: list[str],
    model: sentence_transformers.SentenceTransformer,
    model_uri: str,
    cache_dir: str,
    ignore_cache: bool,
) -> collections.Counter[int]:
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cache_uri = build_cache_uri(corpora_uris=corpora_uris, model_uri=model_uri, cache_dir=cache_dir)

    if not ignore_cache and os.path.exists(cache_uri):
        with open(cache_uri, "r", encoding="utf-8") as f_in:
            token_id_dist = json.load(f_in, object_hook=collections.Counter)

        return token_id_dist

    token_id_dist = collections.Counter()

    for i, duri in enumerate(corpora_uris, 1):
        duri = duri.rstrip("/")
        file_uris = glob.glob(os.path.join(duri, "**", "*.txt"), recursive=True)
        pbar = tqdm.tqdm(
            file_uris, leave=True, desc=f"({i}/{len(corpora_uris)}, {os.path.basename(duri)})"
        )
        for furi in pbar:
            with open(furi, "r", encoding="utf-8") as f_in:
                text = f_in.read()
            token_ids = model.tokenizer.encode(text, add_special_tokens=False)
            token_id_dist.update(token_ids)

    with open(cache_uri, "w", encoding="utf-8") as f_out:
        json.dump(token_id_dist, f_out)

    return token_id_dist


def reduce(
    model_uri: str,
    corpora_uris: list[str],
    new_dict_size: int,
    output_dir: str,
    cache_dir: str,
    word_embedding_submodule_name: str,
    ignore_cache: bool,
    dry_run: bool,
) -> None:
    cache_dir = os.path.abspath(cache_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(cache_dir, exist_ok=True)

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    model_uri = os.path.expanduser(model_uri)
    model_uri = os.path.abspath(model_uri)
    model = sentence_transformers.SentenceTransformer(model_uri, device="cpu")
    model.eval()

    if new_dict_size >= model.tokenizer.vocab_size:
        raise ValueError(
            "New dictionary size is not smaller than current dictionary size "
            f"({new_dict_size=} > {model.tokenizer.vocab_size=})."
        )

    token_id_dist = compute_token_id_dist(
        corpora_uris=corpora_uris,
        model=model,
        model_uri=model_uri,
        cache_dir=cache_dir,
        ignore_cache=ignore_cache,
    )

    special_tokens: list[str] = [
        k for k, v in model.tokenizer.get_vocab().items() if k.startswith("[") and k.endswith("]")
    ]
    special_tokens_token_to_id: dict[str, int] = {
        special_token: model.tokenizer.vocab[special_token] for special_token in special_tokens
    }
    special_tokens_id_to_token: dict[int, str] = {
        v: k for k, v in special_tokens_token_to_id.items()
    }

    with torch.no_grad():
        word_embeddings = model._first_module().get_submodule(word_embedding_submodule_name)
        word_embeddings.requires_grad_(False)

        model_name = f"{os.path.basename(model_uri)}_{new_dict_size}"
        final_output_dir = os.path.join(output_dir, model_name)

        most_common_token_ids: list[int] = sorted(
            [int(k) for k, _ in token_id_dist.most_common(new_dict_size)]
        )

        new_emb = torch.nn.modules.sparse.Embedding(
            num_embeddings=new_dict_size + len(special_tokens),
            embedding_dim=word_embeddings.embedding_dim,
            padding_idx=word_embeddings.padding_idx,
        )
        new_emb.requires_grad_(False)
        new_emb.weight *= 0.0

        new_vocab = {}

        for special_token_id in special_tokens_id_to_token:
            # Keep special tokens in the original index.
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

        temp_file_uri = os.path.join(cache_dir, "new_vocab.temp")
        with open(temp_file_uri, "w", encoding="utf-8") as f_out:
            f_out.write(
                "\n".join([k for k, _ in sorted(new_vocab.items(), key=lambda item: int(item[1]))])
            )

        new_model.tokenizer._tokenizer.model = new_model.tokenizer._tokenizer.model.from_file(
            temp_file_uri
        )

        new_bert = new_model._first_module()
        new_bert.auto_model.resize_token_embeddings(new_emb.num_embeddings)
        new_bert.auto_model.embeddings.word_embeddings = new_emb

        if dry_run:
            return

        new_model.save(path=final_output_dir, model_name=model_name)

        sentence_transformers.SentenceTransformer(final_output_dir, device="cpu")
        os.remove(temp_file_uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_uri", help="SBERT model uri")
    parser.add_argument(
        "corpora_uris",
        help=(
            "One or more corpora directories. This script searches '.txt' files recursively "
            "in the given directories."
        ),
        nargs="+",
        action="extend",
        type=str,
    )

    parser_general = parser.add_argument_group("general arguments")
    parser_general.add_argument(
        "--new-dict-size", "-n", default=30000, type=int, help="New dictionary size"
    )
    parser_general.add_argument(
        "--word-embedding-submodule-name",
        "-s",
        default="auto_model.embeddings.word_embeddings",
        help=(
            "Name of the Torch word embedding submodule, where the (sub)word embeddings will "
            "be retrieved from."
        ),
    )

    parser_output = parser.add_argument_group("output arguments")
    parser_output.add_argument(
        "--output-dir", "-d", default="./reduced_models", help="Specify the output directory."
    )
    parser_output.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, run this script but do not save the output model.",
    )

    parser_cache = parser.add_argument_group("cache arguments")
    parser_cache.add_argument("--cache-dir", default="./cache", help="Specify cache directory.")
    parser_cache.add_argument(
        "--ignore-cache",
        action="store_true",
        help="If set, ignore cache files and recompute token distribution.",
    )

    args = parser.parse_args()
    reduce(**vars(args))
