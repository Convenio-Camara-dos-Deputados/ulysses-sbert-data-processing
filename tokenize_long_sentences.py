import argparse
import functools

import sentence_transformers
import pandas as pd
import numpy as np
import tqdm

import dataset_builder.utils


def run(
    input_uri: str,
    output_uri: str,
    sbert_uri: str,
    context_length: int | None,
    sep: str,
    dry_run: bool,
    it_to_print: int,
) -> None:
    tokenizer = sentence_transformers.SentenceTransformer(sbert_uri, device="cpu").tokenizer
    df = pd.read_csv(input_uri, sep=sep, index_col=0)

    if context_length is None:
        context_length = tokenizer.model_max_length

    pairs: list[tuple[str, str]] = []

    fn_decode = functools.partial(
        tokenizer.decode,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    truncated = 0
    pbar = tqdm.tqdm(df.iterrows(), total=len(df))

    for i, (sent_a, sent_b, source_name) in pbar:
        ids_a = tokenizer.encode(sent_a, truncation=False)
        ids_b = tokenizer.encode(sent_b, truncation=False)

        truncated += int(len(ids_a) > context_length)
        truncated += int(len(ids_b) > context_length)
        pbar.set_description(f"Truncated: {truncated} ({100.0 * truncated / (1 + i):.1f}%)")

        text_a = fn_decode(ids_a[:context_length])
        text_b = text_prev = fn_decode(ids_b[:context_length])
        pairs.append((text_a, text_b, source_name))

        for i_start in np.arange(context_length, len(ids_b), context_length):
            i_end = i_start + context_length
            cur_ids_b = ids_b[i_start:i_end]

            if len(cur_ids_b) <= 32:
                continue

            cur_text_b = fn_decode(cur_ids_b)
            pairs.append((text_prev, cur_text_b, source_name))
            text_prev = cur_text_b

        if pairs and it_to_print and i % it_to_print == 0:
            dataset_builder.utils.print_example(*pairs[-1][:2])

    df_out = pd.DataFrame(pairs, columns=df.columns)

    if not dry_run:
        df_out.to_csv(output_uri, sep=sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_uri", type=str)
    parser.add_argument("output_uri", type=str)
    parser.add_argument("sbert_uri", type=str)
    parser.add_argument("-l", "--context-length", type=int, default=None)
    parser.add_argument("-s", "--sep", type=str, default="\t")
    parser.add_argument("--it-to-print", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(**vars(args))
