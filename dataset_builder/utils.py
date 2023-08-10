import typing as t
import os
import regex as re
import string

import segmentador
import pandas as pd
import numpy as np
import nltk
import colorama


class Config:
    TESEMO_PATH = os.path.abspath("./tesemo_v2.1")
    COMPLEMENTARY_DATADIR = os.path.abspath("./ulysses_sbert_complementary_data")
    IT_TO_PRINT = 0.20


try:
    nltk.data.find("tokenizers/punkt.zip")
except LookupError:
    nltk.download("punkt", quiet=True)

reg_vigorar = re.compile(r"passam?\s+a\s+vigorar", re.IGNORECASE)
reg_dizeres = re.compile(r"seguintes\s*dizeres\s*:\s*$", re.IGNORECASE)
reg_dots = re.compile(r"(?:\.\s*){6,}|…")
reg_split_quotes = re.compile(
    f"(?<=[{string.punctuation}])\s*(?=[\u201C\u2018])|(?<=\.\s*[\u2019\u201D])"
)
reg_h_whitespace_spans = re.compile(r"\h{2,}")
reg_empty_lines = re.compile(r"\n\h*(?=\n)|^\h*\n|\n\h*$")
reg_cliffhanger = re.compile(r":[\s\"'\u201C\u2018\u2019\u201D]*$")
reg_rem_article = re.compile(r"^Art(igo)?\s*\.?\s*1\s*[^a-záéíóúçâêôãẽõü]*", re.IGNORECASE)
reg_not_alphanumeric = re.compile(r"[^a-z0-9çáéíóúâêôãẽõüà]+", re.IGNORECASE)


DEBUG: t.Final[bool] = True
PAIR_COPY_LAWS: list[tuple[str, str]] = []
PAIR_COPY_QUESTIONS: list[tuple[str, str]] = []


def debug_print(*, exit_: bool = False) -> None:
    if not DEBUG:
        raise RuntimeError("DEBUG mode deactivated, and so is this function.")

    for pair in PAIR_COPY_LAWS:
        print_example(*pair)

    for pair in PAIR_COPY_QUESTIONS:
        print_example(*pair)

    print(gen_dataframe(PAIR_COPY_LAWS, allow_empty=True))
    print(gen_dataframe(PAIR_COPY_QUESTIONS, allow_empty=True))

    print("TOTAL PAIRS (ALL)      :", len(PAIR_COPY_QUESTIONS) + len(PAIR_COPY_LAWS))
    print("TOTAL PAIRS (LAWS)     :", len(PAIR_COPY_LAWS))
    print("TOTAL PAIRS (QUESTIONS):", len(PAIR_COPY_QUESTIONS))

    if exit_:
        exit(0)


def inject_negative_pairs(
    df: pd.DataFrame,
    col_ind_to_shuffle: int = 1,
    col_ind_as_ref: int = 0,
    label_col_name: str = "similaridade",
    random_state: int = 178,
) -> pd.DataFrame:
    df_random = df.copy()
    df_random.iloc[:, col_ind_to_shuffle] = (
        df_random.iloc[:, col_ind_to_shuffle]
        .sample(frac=1.0, replace=False, random_state=random_state)
        .values.tolist()
    )

    col_a = df_random.iloc[:, col_ind_as_ref].apply(lambda x: x.lower().strip())
    col_b = df_random.iloc[:, col_ind_to_shuffle].apply(lambda x: x.lower().strip())
    df_random = df_random.loc[col_a != col_b, :]

    df[label_col_name] = 1
    df_random[label_col_name] = 0

    df_out = pd.concat((df, df_random))
    df_out.reset_index(drop=True, inplace=True)

    return df_out


def to_tsv(df: pd.DataFrame, output_name: str, output_dir: str = "./processed_data") -> None:
    if not output_name.endswith(".tsv"):
        output_name += ".tsv"

    output_dir = os.path.abspath(output_dir)
    output_uri = os.path.join(output_dir, output_name)

    df.to_csv(output_uri, sep="\t")


def natural_sentence_tokenize(text: str, preprocess_whitespaces: bool = True) -> list[str]:
    tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
    tokenizer._params.abbrev_types.update(
        {
            "art",
            "arts",
            "profa",
            "profᵃ",
            "dep",
            "sr",
            "sra",
            "srᵃ",
            "s.exª",
            "s.exa",
            "v.a",
            "v.em.ª",
            "v.ex.ª",
            "v.mag.ª",
            "v.em.a",
            "v.ex.a",
            "v.mag.a",
            "v.m",
            "v.ex.ªrev.ma",
            "v.p",
            "v.rev.ª",
            "v.rev.a",
            "v.rev.ma",
            "v.m.cê",
            "v.s",
            "v.s.ª",
            "v.s.a",
            "v.a",
            "v.emª",
            "v.exª",
            "v.ema",
            "v.exa",
            "v.magª",
            "me",
            "ma",
            "v.sa",
            "v.m",
            "v.ex.ªrev.ma",
            "v.ex.arev.ma",
            "v.p",
            "v.revª",
            "v.rev.ma",
            "v.m.cê",
            "v.s",
            "v.sª",
            "dra",
            "drª",
            "profa",
            "profª",
            "ass",
            "obs",
            "art",
            "par",
            "pag",
            "pág",
            "pars",
            "pags",
            "págs",
            "cap",
            "caps",
            "1º",
            "2º",
            "3º",
            "4º",
            "5º",
            "6º",
            "7º",
            "8º",
            "9º",
            "1o",
            "2o",
            "3o",
            "4o",
            "5o",
            "6o",
            "7o",
            "8o",
            "9o",
            "10o",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "ph.d",
            "phd",
            "et al",
            "adv.º",
            "advº",
            "advª",
            "ind",
            "vol",
            "n",
            "nº",
            "núm",
            "nro",
            "nrº",
            "gab",
            "ex",
            "www",
            "gov",
            "http",
            "https",
            "com",
            "br",
            "org",
            "http://www",
            "https://www",
            "gov.br",
            "org.br",
            "leg",
            "http://www1",
            "https://www1",
            "parág",
            "m.d",
        }
    )

    if preprocess_whitespaces:
        text = remove_spurious_whitespaces(text)

    segs = tokenizer.tokenize(text)
    out = []

    for item in segs:
        out.extend([x.strip() for x in reg_split_quotes.split(item) if x.strip()])

    return out


def remove_spurious_whitespaces(text: str) -> str:
    if not text:
        return text

    text = text.strip()
    text = re.sub(r"(?<=[(\[{/\\\u201C\u2018])\s+|\s+(?=[:?!,.;)\]}/\\\u2019\u201D])", "", text)
    text = re.sub(r"(?<=[a-zàáéíóúãẽõâêôüç])\s*-\s*(?=[a-zàáéíóúãẽõâêôüç])", "-", text)
    text = re.sub(r"(?<=[0-9])\s*\.\s*(?=[0-9])", ".", text)
    text = re.sub(r"\h{2,}", " ", text)

    return text


def gen_dataframe(
    pairs: list[tuple[str, str]],
    source_name: t.Optional[str] = None,
    target: t.Optional[int] = None,
    min_length: int = 30,
    max_length: int = 1200,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not allow_empty and not len(pairs):
        raise ValueError(f"Empty DataFrame provided with '{allow_empty=}'.")

    df = pd.DataFrame(pairs, columns=["sentence_a", "sentence_b"])

    df.dropna(inplace=True)

    df = df.applymap(lambda x: x.strip())
    df = df.applymap(lambda x: x.replace("\r", " "))
    df = df.applymap(lambda x: reg_h_whitespace_spans.sub(" ", x))
    df = df.applymap(lambda x: reg_empty_lines.sub("", x))

    df.drop_duplicates(inplace=True)
    df = df.loc[
        df["sentence_a"].apply(str.lower).values != df["sentence_b"].apply(str.lower).values,
        :,
    ]

    lens = df.applymap(len).min(axis=1).values
    df.loc[lens > max_length, :] = (
        df.loc[lens > max_length, :].applymap(lambda x: f"{x[:max_length-3]}...").values
    )

    df = df.loc[min_length <= lens, :]

    if source_name is not None:
        source_name = _prepare_source_name(source_name)
        df["source"] = source_name

    if target is not None:
        df["label"] = target

    df.reset_index(drop=True, inplace=True)

    return df


def _prepare_source_name(source_name: str) -> str:
    return reg_not_alphanumeric.sub("_", source_name.strip().upper())


def fetch_further_leg_context(
    segs: list[str], start_i: int = 0, min_seg_length: int = 20
) -> tuple[t.Optional[str], int]:
    if start_i >= len(segs):
        return None, start_i

    first_seg = segs[start_i]
    items = [first_seg]
    i = start_i + 1

    if (
        reg_vigorar.search(first_seg)
        or reg_dizeres.search(first_seg)
        or len(first_seg) < min_seg_length
        or reg_cliffhanger.search(first_seg)
    ) and i < len(segs):
        items.append(segs[i])
        i += 1
        while i < len(segs) and (
            len(items[-1]) < min_seg_length
            or reg_cliffhanger.search(items[-1])
            or reg_dots.search(items[-1])
        ):
            items.append(segs[i])
            i += 1

    return ("\n".join([reg_dots.sub(" (...) ", item.strip()) for item in items]), i)


def print_example(seg_a: str, seg_b: str) -> None:
    print(128 * "=")
    print(f">>> {colorama.Fore.YELLOW}{seg_a}{colorama.Style.RESET_ALL}")
    print(128 * "-")
    print(f">>> {colorama.Fore.YELLOW}{seg_b}{colorama.Style.RESET_ALL}")
    print(128 * "=", end="\n\n")


def fetch_first_item_index(
    content: t.Union[str, t.Sequence[str]],
    segmenter: t.Optional[segmentador.Segmenter] = None,
    max_chars: t.Optional[int] = 4000,
    prefix: str = "Art",
) -> int:
    if isinstance(content, str):
        if max_chars is not None:
            content = content[:max_chars]
        segs = segmenter(content, remove_noise_subsegments=True)

    else:
        segs = content

    for i, seg in enumerate(segs):
        if seg.startswith(prefix):
            return i

    return -1


def fetch_first_item_with_index(
    content: t.Union[str, t.Sequence[str]],
    segmenter: t.Optional[segmentador.Segmenter] = None,
    max_chars: t.Optional[int] = 4000,
    prefix: str = "Art",
) -> t.Tuple[t.Optional[str], int]:
    if isinstance(content, str):
        if max_chars is not None:
            content = content[:max_chars]
        segs = segmenter(content, remove_noise_subsegments=True)

    else:
        segs = content

    if not segs:
        return None, -1

    for i, seg in enumerate(segs):
        if seg.startswith(prefix):
            break
    else:
        return None, -1

    content, _ = fetch_further_leg_context(segs, start_i=i)
    content = content.strip()

    if prefix == "Art":
        return reg_rem_article.sub("", content), i

    return content.lstrip(prefix), i


def fetch_first_item(*args: any, **kwargs: any) -> t.Optional[str]:
    return fetch_first_item_with_index(*args, **kwargs)[0]


def fetch_laws_in_segments(
    segs: list[str],
    refs_i: t.Optional[t.Union[str, t.Sequence[int]]] = (0,),
    start_i: int = -1,
    pattern: str = (
        r"\b(?:"
        r"Instrução Normativa|"
        r"IN|"
        r"Medida\s*Provisória|"
        r"MP|"
        r"Lei\s*Complementar|"
        r"Lei|"
        r"PL|"
        r"PLC|"
        r"Decreto|"
        r"Decreto-Lei|"
        r"PEC|"
        r"CF|"
        r"Constituição\s*Federal"
        r")\b.{,25}[\.0-9]{2,}|"
        r"\b(?:Código|Estatuto\s*d[oae]|Lei\s+(?:d[oae]|[A-ZÇÀÁÉÍÓÚÂÊÔÃẼÕÜ]))\b"
    ),
    min_ref_len: int = 48,
    min_law_len: int = 64,
) -> list[tuple[str, str]]:
    if refs_i and not isinstance(refs_i, str):
        refs_i = [i for i in refs_i if i < len(segs) and len(segs[i]) >= min_ref_len]
        if not refs_i:
            return []

        if start_i < 0 or start_i <= max(refs_i):
            start_i = max(refs_i) + 1

    reg_laws = re.compile(pattern)
    new_pairs: list[tuple[str, str]] = []

    start_i = max(0, start_i)
    k = 0

    for j in range(start_i, len(segs)):
        if reg_laws.search(segs[j]) and len(segs[j]) >= min_law_len:
            if refs_i:
                if isinstance(refs_i, str):
                    new_pairs.append((refs_i, segs[j]))

                else:
                    new_pairs.append((segs[refs_i[k]], segs[j]))
                    k = (k + 1) % len(refs_i)

                continue

            subsegs = natural_sentence_tokenize(segs[j])
            m = len(subsegs) // 2
            a, b = " ".join(subsegs[:m]), " ".join(subsegs[m:])
            if min(len(a), len(b)) >= min_law_len:
                new_pairs.append((a, b))

    if DEBUG:
        PAIR_COPY_LAWS.extend(new_pairs)

    return new_pairs


def fetch_questions_in_segments(
    segs: list[str],
    start_i: int = 0,
    context_i: t.Optional[int] = None,
    pattern: str = r".{50,300}\?\s*$",
    min_context_len: int = 48,
) -> list[tuple[str, str]]:
    reg_questions = re.compile(pattern)

    new_pairs: list[tuple[str, str]] = []
    context = ""

    if context_i is not None:
        while context_i < len(segs) and len(segs[context_i]) < min_context_len:
            context_i += 1

        if context_i < len(segs):
            subsegs = natural_sentence_tokenize(segs[context_i])

            k = 1
            total_len = len(subsegs[0]) if subsegs else 0

            while k < len(subsegs) and total_len < min_context_len:
                total_len += len(subsegs[k])
                k += 1

            context = " ".join(subsegs[:k])
            context = f"({context}) "

    j = start_i
    while j < len(segs) - 1:
        if (
            reg_questions.match(segs[j])
            and len(segs[j + 1]) > 64
            and "?" not in segs[j + 1].rstrip()[-2:]
        ):
            new_pairs.append((f"{context}{segs[j]}", segs[j + 1]))
            j += 1

        j += 1

    if DEBUG:
        PAIR_COPY_QUESTIONS.extend(new_pairs)

    return new_pairs
