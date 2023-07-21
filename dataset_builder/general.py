import typing as t
import glob
import os
import re
import collections

import pandas as pd
import numpy as np
import tqdm

from . import utils


def _read_file_segments(
    uri: str,
    /,
    *,
    min_seg_len: int = 1,
    redundancy_check_inds: tuple[int, int] | None = None,
    reg_banned_patterns: re.Pattern | None = None,
    fn_seg_preproc: t.Callable[[str], str] | None = None,
    full_search_banned_patterns: bool = False,
) -> list[str]:
    with open(uri, "r", encoding="utf-8") as f_in:
        segs = f_in.readlines()

    segs = [item.strip() for item in segs]

    if redundancy_check_inds:
        i, j = redundancy_check_inds
        if len(segs) > max(i, j) and segs[i].lower() == segs[j].lower():
            segs.pop(i)

    segs = [item for item in segs if len(item) >= min_seg_len]

    if reg_banned_patterns:
        fn_filter = reg_banned_patterns.search if full_search_banned_patterns else reg_banned_patterns.match
        segs = [item for item in segs if not fn_filter(item)]

    if fn_seg_preproc:
        segs = [fn_seg_preproc(item) for item in segs]

    return segs


def _make_pairs_generic(
    uri: str,
    /,
    *,
    source_name: str,
    long_segments: bool,
    long_segment_inds: tuple[int | slice, int | slice, int],
    short_segment_inds: list[tuple[int, int, int]],
    fetch_law_in_segments: bool,
    fetch_questions_in_segments: bool,
    min_seg_len: int = 1,
    redundancy_check_inds: tuple[int, int] | None = None,
    match_reg_banned_patterns: re.Pattern | None = None,
    search_reg_banned_patterns: re.Pattern | None = None,
    reg_document_full_skip: re.Pattern | None = None,
    fn_seg_preproc: t.Callable[[str], str] | None = None,
    document_full_skip_inds: list[int] | None = None,
    it_to_print: int | None = None,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    source_dir = os.path.join(utils.TESEMO_PATH, uri, "*.txt")
    n = it_to_print or float("+inf")

    for uri in tqdm.tqdm(glob.glob(source_dir), desc=source_name):
        n -= 1

        segs = _read_file_segments(
            uri,
            min_seg_len=min_seg_len,
            match_reg_banned_patterns=match_reg_banned_patterns,
            search_reg_banned_patterns=search_reg_banned_patterns,
            redundancy_check_inds=redundancy_check_inds,
            fn_seg_preproc=fn_seg_preproc,
        )

        skip_document = False

        if reg_document_full_skip:
            for i in document_full_skip_inds:
                if len(segs) > i and reg_document_full_skip.search(segs[i]):
                    skip_document = True
                    break

        if skip_document:
            continue

        if long_segments:
            i, j, min_len = long_segment_inds

            max_ind = max(
                int(i.stop) if isinstance(i, slice) else (i + 1),
                int(j.stop) if isinstance(j, slice) else (j + 1),
            )

            if len(segs) >= max_ind:
                seg_a = "\n".join(segs[i]) if isinstance(i, slice) else segs[i]
                seg_b = "\n".join(segs[j]) if isinstance(j, slice) else segs[j]

                if min(len(seg_a), len(seg_b)) >= min_len:
                    pairs.append((seg_a, seg_b))

        else:
            for i, j, min_len in short_segment_inds:
                if len(segs) > max(i, j):
                    seg_a, seg_b = segs[i], segs[j]
                    if min(len(seg_a), len(seg_b)) >= min_len:
                        pairs.append((seg_a, seg_b))

            if fetch_law_in_segments:
                pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=[0, 2]))

            if fetch_questions_in_segments:
                pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=0))

        if n <= 0:
            n = it_to_print
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_ministerios(*, long_segments: bool) -> dict[str, list[tuple[str, str]]]:
    pairs = collections.defaultdict(list)

    reg_ministerio = re.compile(r"https[^a-z]+www[^a-z]+gov[^a-z]+br[^a-z]+([a-z]+)", re.IGNORECASE)
    reg_foto = re.compile(r"[^\.a-zç]*\s*Fotos?:.+$", re.IGNORECASE)
    base_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/ministerios")

    for uri in tqdm.tqdm(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n")]

        if len(segs) < 3:
            continue

        ministerio = os.path.basename(os.path.dirname(uri))

        if os.path.basename(uri).startswith("http"):
            if long_segments:
                if len(segs) >= 2 and len(segs[0]) >= 48:
                    pairs[ministerio].append((segs[0], "\n".join(segs[1:])))

            else:
                if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 48:
                    pairs[ministerio].append((segs[0], segs[1]))

                if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                    pairs[ministerio].append((segs[2], segs[1]))

                pairs[ministerio].extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
                pairs[ministerio].extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        else:
            if long_segments:
                if len(segs) >= 8:
                    seg_a = f"{segs[0]}\n{segs[1]}\n{segs[2]}"
                    seg_b = "\n".join(segs[8:])

                    if min(len(seg_a), len(seg_b)) >= 48:
                        pairs[ministerio].append((seg_a, seg_b))

            else:
                if min(len(segs[1]), len(segs[2])) >= 48:
                    pairs[ministerio].append((segs[1], segs[2]))

                pairs[ministerio].extend(utils.fetch_laws_in_segments(segs=segs, refs_i=[1, 2]))
                pairs[ministerio].extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=1))

                if len(segs) >= 10:
                    segs[8] = reg_foto.sub("", segs[8])

                    if min(len(segs[8]), len(segs[9])) >= 48:
                        pairs[ministerio].append((segs[8], segs[9]))

    return pairs


def make_pairs_tv_camara(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(r"Créditos?")
    reg_full_document_skip = re.compile(
        r"Revista da Câmara|"
        r"Veja o que|"
        r"Câmara Hoje|"
        r"agenda da Câmara|"
        r"Veja os destaques|"
        r"desta semana|"
        r"nesta semana|"
        r"semana passada|"
        r"na Câmara hoje|"
        r"Principais notícias"
        "",
        re.IGNORECASE,
    )

    for uri in tqdm.tqdm(glob.glob(os.path.join(utils.TESEMO_PATH, "outros/o8_tv_camara/*.txt"))):
        with open(uri, "r") as f_in:
            segs = [
                item.strip() for item in f_in.read().strip().split("\n") if item.strip() and not reg_banned_patterns.match(item)
            ]

        if (
            len(segs) < 2
            or reg_full_document_skip.search(segs[1])
            or (len(segs) >= 3 and reg_full_document_skip.search(segs[2]))
        ):
            continue

        if long_segments:
            if len(segs) >= 3 and len(segs[1]) >= 60:
                pairs.append((segs[1], "\n".join(segs[2:])))

        else:
            if len(segs) >= 3 and min(len(segs[1]), len(segs[2])) >= 60:
                pairs.append((segs[1], segs[2]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, refs_i=[1, 2]))

    assert len(pairs)
    return pairs


def make_pairs_radio_e_tv_justica() -> list[tuple[str, str]]:
    df = pd.read_csv(
        os.path.join(utils.COMPLEMENTARY_DATADIR, "radio_e_tv_justica_ementas.tsv"),
        sep="\t",
        index_col=0,
    )
    pairs = df.values.tolist()
    assert len(pairs)
    return pairs


def make_pairs_mpt(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for uri in glob.glob(
        os.path.join(
            utils.TESEMO_PATH,
            "outros/o1_noticias_governamentais/mpt_ministerio_publico_do_trabalho/*.txt",
        )
    ):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.readlines() if item.strip()]

        if long_segments:
            if len(segs) >= 3 and len(segs[0]) >= 40:
                pairs.append((segs[0], "\n".join(segs[2:])))

        else:
            if len(segs) >= 3 and min(len(segs[0]), len(segs[2])) >= 40:
                pairs.append((segs[0], segs[2]))

            if len(segs) >= 4 and min(len(segs[0]), len(segs[3])) >= 40:
                pairs.append((segs[0], segs[3]))

            if len(segs) >= 5 and min(len(segs[4]), len(segs[3])) >= 40:
                pairs.append((segs[4], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=[0, 2]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=0))

    assert len(pairs)
    return pairs


def make_pairs_mpm(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for uri in glob.glob(
        os.path.join(
            utils.TESEMO_PATH,
            "outros/o1_noticias_governamentais/mpm_ministerio_publico_militar/*.txt",
        )
    ):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.readlines() if item.strip()]

        if long_segments:
            if len(segs) >= 3 and len(segs[1]) >= 40:
                pairs.append((segs[1], "\n".join(segs[2:])))

        else:
            if len(segs) >= 3 and min(len(segs[1]), len(segs[2])) >= 40:
                pairs.append((segs[1], segs[2]))

            if len(segs) >= 4 and min(len(segs[3]), len(segs[2])) >= 40:
                pairs.append((segs[3], segs[2]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=4, refs_i=[1, 2]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=1))

    assert len(pairs)
    return pairs


def make_pairs_tcu(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for uri in glob.glob(
        os.path.join(
            utils.TESEMO_PATH,
            "outros/o1_noticias_governamentais/tcu_tribunal_de_contas_da_uniao/*.txt",
        )
    ):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.readlines() if len(item.strip()) >= 40]

        if not segs or "Destaques da sessão plenária" in segs[0]:
            continue

        if long_segments:
            if len(segs) >= 2:
                pairs.append(("\n".join(segs[:2]), "\n".join(segs[2:])))

        else:
            if len(segs) >= 2:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3:
                pairs.append((segs[0], segs[2]))

            if len(segs) >= 4:
                pairs.append((segs[3], segs[2]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=4, refs_i=[0, 1, 2]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=0))

    assert len(pairs)
    return pairs


def make_pairs_radio_camara(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg = re.compile(
        r"(?:loc-|boa noite|est[aá] no ar|Edição [^0-9a-zç]|(?:primeiro|segundo) bloco|bloco [0-9]|[aáà] seguir)",
        re.IGNORECASE,
    )

    for uri in tqdm.tqdm(
        glob.glob(os.path.join(utils.TESEMO_PATH, "outros/o3_radio_camara/*.txt")),
        desc="radio_camara",
    ):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if len(item.strip()) >= 100]

        if long_segments:
            if len(segs) >= 2:
                seg_a = segs[0]
                seg_b = "\n".join(segs[1:])
                if not reg.search(seg_a) and not reg.search(seg_b):
                    pairs.append((seg_a, seg_b))

        else:
            if (
                len(segs) >= 2
                and max(len(segs[0]), len(segs[1])) <= 700
                and not reg.search(segs[0])
                and not reg.search(segs[1])
            ):
                pairs.append((segs[0], segs[1]))

    assert len(pairs)
    return pairs


def make_pairs_trf4(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf4_tribunal_regional_federal_da_4_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc="news_trf4"):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if len(segs) == 0 or "Agenda da presidente do TRF4" in segs[0]:
            continue

        if long_segments:
            if len(segs) >= 5 and len(segs[1]) >= 64:
                pairs.append((segs[1], "\n".join(segs[4:])))

        else:
            if len(segs) >= 5 and min(len(segs[1]), len(segs[4])) >= 64:
                pairs.append((segs[1], segs[4]))

            if len(segs) >= 6 and min(len(segs[5]), len(segs[4])) >= 64:
                pairs.append((segs[5], segs[4]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=6, refs_i=[1, 4]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=5, context_i=1))

    assert len(pairs)
    return pairs


def make_pairs_trf3(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf3_tribunal_regional_federal_da_3_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir)):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if long_segments:
            segs = [item for item in segs if len(item) >= 48]

            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                pairs.append((seg_a, seg_b))

            elif len(segs) == 2:
                pairs.append((segs[0], segs[1]))

        else:
            if os.path.basename(uri).startswith("http"):
                if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 48:
                    pairs.append((segs[0], segs[1]))

                if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 64:
                    pairs.append((segs[2], segs[1]))

                pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
                pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

            else:
                if len(segs) >= 8 and min(len(segs[5]), len(segs[7])) >= 48:
                    pairs.append((segs[5], segs[7]))

                if len(segs) >= 9 and min(len(segs[6]), len(segs[8])) >= 48:
                    pairs.append((segs[6], segs[8]))

                pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=9, refs_i=[5, 7]))
                pairs.extend(utils.fetch_questions_in_segments(segs, start_i=8, context_i=5))

    assert len(pairs)
    return pairs


def make_pairs_trf2(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf2_tribunal_regional_federal_da_2_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir)):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if long_segments:
            if os.path.basename(uri).startswith("http"):
                if len(segs) >= 4:
                    seg_a = "\n".join([segs[0], segs[2]])
                    seg_b = "\n".join(segs[3:])
                    if min(len(seg_a), len(seg_b)) >= 64:
                        pairs.append((seg_a, seg_b))

            else:
                segs = [item.rstrip("*").strip() for item in segs]
                if len(segs) >= 5:
                    seg_a = "\n".join([segs[1], segs[3]])
                    seg_b = "\n".join(segs[4:])
                    if min(len(seg_a), len(seg_b)) >= 64:
                        pairs.append((seg_a, seg_b))

        else:
            if os.path.basename(uri).startswith("http"):
                if len(segs) >= 3 and min(len(segs[0]), len(segs[2])) >= 48:
                    pairs.append((segs[0], segs[2]))

                if len(segs) >= 4 and min(len(segs[3]), len(segs[2])) >= 64:
                    pairs.append((segs[3], segs[2]))

                pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=4, refs_i=[0, 2]))
                pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=0))

            else:
                segs = [item.rstrip("*").strip() for item in segs]

                if len(segs) >= 4 and min(len(segs[1]), len(segs[3])) >= 48:
                    pairs.append((segs[1], segs[3]))

                if len(segs) >= 5 and min(len(segs[4]), len(segs[3])) >= 64:
                    pairs.append((segs[4], segs[3]))

                pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=[1, 3]))
                pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=1))

    assert len(pairs)
    return pairs


def make_pairs_trf1(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg = re.compile(
        r"(?:\n|\d\d/\d\d/\d\d \d\d:\d\d|Cr[eé]dito: (?:imagem da web|Google imagens|Reprodução/Internet|Internet))",
        re.IGNORECASE,
    )
    reg_noise_patterns = re.compile(r"(?:DECISÃO|INSTITUCIONAL):\s*")
    reg_banned_patterns = re.compile(r"Créditos?:")

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf1_tribunal_regional_federal_da_1_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir)):
        with open(uri, "r") as f_in:
            segs = [reg_noise_patterns.sub("", item.strip()) for item in reg.split(f_in.read().strip())]
            segs = [item for item in segs if len(item) >= 48 and not reg_banned_patterns.match(item)]

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 48:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 64:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

    assert len(pairs)
    return pairs


def make_pairs_stj(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(r"Atualizad[oa] em")
    reg_date = re.compile(r"[0-9]{2}/[0-9]{2}")

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/stj_superior_tribunal_de_justica/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir)):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n")]
            segs = [item for item in segs if len(item) >= 5 and not reg_banned_patterns.match(item)]

        if reg_date.match(segs[1]):
            segs.pop(1)

        if long_segments:
            segs = utils.natural_sentence_tokenize(" ".join(segs[1:]))
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 3 and min(len(segs[1]), len(segs[2])) >= 48:
                pairs.append((segs[1], segs[2]))

            if len(segs) >= 4 and min(len(segs[3]), len(segs[2])) >= 64:
                pairs.append((segs[3], segs[2]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=[2, 3]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=2))

    assert len(pairs)
    return pairs


def make_pairs_tse(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/tse_tribunal_superior_eleitoral/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir)):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if long_segments:
            if len(segs) >= 4:
                seg_a = "\n".join([segs[0], segs[1]])
                seg_b = "\n".join(segs[3:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 4 and min(len(segs[0]), len(segs[3])) >= 48:
                pairs.append((segs[0], segs[3]))

            if len(segs) >= 4 and min(len(segs[1]), len(segs[3])) >= 48:
                pairs.append((segs[1], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=4, refs_i=[0, 3]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=0))

    assert len(pairs)
    return pairs


def make_pairs_cnmp(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/cnmp_conselho_nacional_do_ministerio_publico/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc="news_cnmp"):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if long_segments:
            if len(segs) >= 4:
                seg_a = "\n".join(segs[2:4])
                seg_b = "\n".join(segs[4:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 4 and min(len(segs[2]), len(segs[3])) >= 48:
                pairs.append((segs[2], segs[3]))

            if len(segs) >= 5 and min(len(segs[4]), len(segs[3])) >= 48:
                pairs.append((segs[4], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=[2, 3]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=2))

    assert len(pairs)
    return pairs


def make_pairs_bc(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/banco_central/*.txt")
    for uri in tqdm.tqdm(glob.glob(source_dir), desc="news_bc"):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if item.strip()]

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 64:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[0]), len(segs[2])) >= 64:
                pairs.append((segs[0], segs[2]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

    assert len(pairs)
    return pairs


def make_pairs_camara_noticias_comments() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg = re.compile(r"https_www_camara_leg_br_noticias_\d+_")
    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/comentarios_noticias_camara_dos_deputados/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc="news_camara_comments"):
        with open(uri, "r") as f_in:
            segs = f_in.read().strip().split("\n")

        split_1 = reg.sub("", os.path.basename(uri)).replace(".txt", "").replace("_", " ")

        for split_2 in segs:
            if len(split_2) >= 30:
                pairs.append((split_1, split_2))

    assert len(pairs)
    return pairs


def make_pairs_camara_noticias(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    source_dir = glob.glob(
        os.path.join(
            utils.TESEMO_PATH,
            "outros/o1_noticias_governamentais/camara_dos_deputados/*.txt",
        )
    )
    for uri in tqdm.tqdm(source_dir, desc="news_camara"):
        with open(uri, "r") as f_in:
            segs = [item.strip() for item in f_in.read().strip().split("\n") if len(item.strip()) >= 25]

        if long_segments:
            if len(segs) >= 3:
                seg_a = segs[1]
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 3 and min(len(segs[1]), len(segs[2])) >= 48:
                pairs.append((segs[1], segs[2]))

            if len(segs) >= 4 and min(len(segs[1]), len(segs[3])) >= 48:
                pairs.append((segs[1], segs[3]))

            if len(segs) >= 5 and min(len(segs[4]), len(segs[3])) >= 48:
                pairs.append((segs[4], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=5, refs_i=None))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=4, context_i=1))

    assert len(pairs)
    return pairs


def make_pairs_senado_noticias(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_split = re.compile(
        r"^\s*(?:Saiba mais|Buscar|Mais vídeos|MAIS NOTÍCIAS SOBRE|Mais informações a seguir|Mais vistas|Últimas:)$",
        re.IGNORECASE | re.MULTILINE,
    )
    reg_banned_patterns = re.compile(
        r"^Compartilhe este conteúdo no|"
        r"^Home|"
        r"^Agência Senado \(Reprodução autorizada mediante citação da Agência Senado\)|"
        r"^Compartilhar:|"
        r"^Compartilhe:|"
        r"[0-9]{2}/[0-9]{2}/[0-9]{4}\s*,\s+[0-9]{2}[:h][0-9]{2}|"
        r"^O projeto foi relatado.{,60}$"
        "",
        re.IGNORECASE,
    )

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/senado_federal/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_senado")):
        if "cpi" in uri:
            continue

        with open(uri, "r") as f_in:
            segs = reg_split.split(f_in.read())[0].split("\n")
            segs = [item.strip() for item in segs]
            segs = [item for item in segs if len(item) >= 30 and not reg_banned_patterns.search(item)]

        if len(segs) >= 2 and segs[0].lower() == segs[1].lower():
            segs.pop(0)

        if long_segments:
            if len(segs) >= 2:
                seg_a = segs[0]
                seg_b = "\n".join(segs[1:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 48:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 64:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=None))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        if i % 5000 == 0 and pairs:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_stf(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"^.{,80}atualizado há|"
        r"pessoas já viram isso|"
        r"Pauta de julgamentos previstos|"
        r"Confira, abaixo, o resumo dos julgamentos previstos"
        r"",
        re.IGNORECASE,
    )
    reg_full_document_skip = re.compile(
        r"Agenda do presidente|"
        r"Confira os destaques da TV Justiça|"
        r"Destaques da Rádio Justiça|"
        r"Confira a programação"
        r"",
        re.IGNORECASE,
    )

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/stf_superior_tribunal_federal/*.txt",
    )
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_stf")):
        with open(uri, "r") as f_in:
            segs = [
                item.strip()
                for item in f_in.read().strip().split("\n")
                if item.strip() and not reg_banned_patterns.search(item)
            ]

        if not segs or reg_full_document_skip.search(segs[0]):
            continue

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 32:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 32:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=None))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        if i % 5000 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_stm(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"\d{2}/\d{2}/\d{4}$|"
        r"Imprimir\s*E-mail\s*|"
        r"Imagem ilustrativa|"
        r"foto: .{,80}$|"
        r"Conselho Nacional de Justiça$|"
        r"Supremo Tribunal Federal$|"
        r"Crédito: .{,80}$"
        r"",
        re.IGNORECASE,
    )

    source_dir = os.path.join(
        utils.TESEMO_PATH,
        "outros/o1_noticias_governamentais/stm_superior_tribunal_militar/*.txt",
    )
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_stm")):
        with open(uri, "r") as f_in:
            segs = [
                item.strip()
                for item in f_in.read().strip().split("\n")
                if item.strip() and len(item.strip()) >= 32 and not reg_banned_patterns.match(item.strip())
            ]

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 32:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 32:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        if i % 5000 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_tst(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"Esse conteúdo não está disponível sem cookies|"
        r"null$|"
        r"imprimir$|"
        r"Notícias do TST$|"
        r"Seguir$|"
        r"\([STQD]"
        r"",
        re.IGNORECASE,
    )

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/tst_tribunal_superior_do_trabalho/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_tst")):
        with open(uri, "r") as f_in:
            segs = [
                item.strip()
                for item in f_in.read().strip().split("\n")
                if item.strip() and not reg_banned_patterns.match(item.strip())
            ]

        if segs[0] == segs[1]:
            segs.pop(0)

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 32:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 32:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        if i % 5000 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_tst_radio(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_full_document_skip = re.compile(r"Trabalho e Justiça", re.IGNORECASE)

    reg_banned_patterns = re.compile(
        r"^null$|"
        r"^\d{2}.\d{2}.\d{4}\s*.{,5}$|"
        r"Baixe o [aá]udio|"
        r"Reproduzir o áudio|"
        r"O programa Trabalho e Justiça vai ao ar|"
        r"Ouça os detalhes na reportagem|"
        r"Permitida a reprodução mediante citação da fonte|"
        r"Esta matéria tem caráter informativo, sem cunho oficial|"
        r"Esse conteúdo não está disponível sem cookies|"
        r"E também acompanha|"
        r"E mais:|"
        r"Você também vai saber|"
        r"Você também vai var|"
        r"Rádio Justiça|"
        r"^Processo:|"
        r"^Mais detalhes$|"
        r"^Para saber mais, aperte o play!|"
        r"\([STQD]"
        r"",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile(r"^REPÓRTER:\s*|\s*\|\s*TST na Voz do Brasil", re.IGNORECASE)

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o7_radio_tst/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="radio_tst")):
        with open(uri, "r") as f_in:
            segs = [
                item.strip()
                for item in f_in.read().strip().split("\n")
                if item.strip() and not reg_banned_patterns.search(item.strip())
            ]

        if len(segs) >= 1 and reg_full_document_skip.search(segs[0]):
            continue

        if long_segments:
            if len(segs) >= 2:
                seg_a = segs[0]
                seg_b = "\n".join(segs[1:])
                if min(len(seg_a), len(seg_b)) >= 32:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 32:
                segs[0] = reg_noise_patterns.sub("", segs[0])
                segs[1] = reg_noise_patterns.sub("", segs[1])
                pairs.append((segs[0], segs[1]))

        if i % 500 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_tst_tv(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"^null$|"
        r"^\s*Esse conteúdo não está disponível sem cookies|"
        r"E também acompanha|"
        r"Você também vai saber|"
        r"você também vai ver|"
        r"^\s*\(.{,45}\d{2,4}\)\s*$|"
        r"^Mais detalhes$|"
        r"^Processo:|"
        r"^\d{2}/\d{2}/\d{2,4}\s*.?$",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile(r"\| Programa completo", re.IGNORECASE)

    reg_skip = re.compile(
        r"Nesta edição você confere|Confira nessa edição|Veja os destaques desta edição|Nesta edição você confere|Veja os destaques desta edição",
        re.IGNORECASE,
    )

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o10_tst_tv/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="tv_tst")):
        with open(uri, "r") as f_in:
            segs = [
                item.strip()
                for item in f_in.read().strip().split("\n")
                if item.strip() and len(item.strip()) >= 20 and not reg_banned_patterns.search(item.strip())
            ]

        if not segs or reg_skip.search(segs[0]):
            continue

        if long_segments:
            if len(segs) >= 2:
                seg_a = segs[0]
                seg_b = "\n".join(segs[1:])
                if min(len(seg_a), len(seg_b)) >= 21:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 21:
                segs[0] = reg_noise_patterns.sub("", segs[0])
                segs[1] = reg_noise_patterns.sub("", segs[1])
                pairs.append((segs[0], segs[1]))

        if i % 500 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_trf6(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_noise_patterns = re.compile(r"Início\s*»\s*", re.IGNORECASE)

    source_dir = os.path.join(
        utils.TESEMO_PATH, "outros/o1_noticias_governamentais/trf6_tribunal_regional_federal_da_6_regiao/*.txt"
    )
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_trf6")):
        with open(uri, "r") as f_in:
            segs = [
                reg_noise_patterns.sub("", item.strip())
                for item in f_in.read().strip().split("\n")
                if item.strip() and len(item.strip()) >= 48
            ]

        if segs[0] == segs[1]:
            segs.pop(0)

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 48:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                pairs.append((segs[2], segs[1]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=3, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=2, context_i=0))

        if i % 20 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_trf5(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_noise_patterns = re.compile(r"Início\s*»\s*", re.IGNORECASE)
    reg_banned_patterns = re.compile("Última atualização:|^Por:", re.IGNORECASE)

    source_dir = os.path.join(
        utils.TESEMO_PATH, "outros/o1_noticias_governamentais/trf5_tribunal_regional_federal_da_5_regiao/*.txt"
    )
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_trf5")):
        with open(uri, "r") as f_in:
            aux = [
                reg_noise_patterns.sub("", item.strip())
                for item in f_in.read().strip().split("\n")
                if item.strip() and len(item.strip()) >= 32 and not reg_banned_patterns.search(item)
            ]

            segs = []
            for item in aux:
                segs.extend(utils.natural_sentence_tokenize(item))

        if long_segments:
            if len(segs) >= 3:
                seg_a = "\n".join(segs[:2])
                seg_b = "\n".join(segs[2:])
                if min(len(seg_a), len(seg_b)) >= 48:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 32:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 48:
                pairs.append((segs[2], segs[1]))

            if len(segs) >= 4 and min(len(segs[2]), len(segs[3])) >= 160:
                pairs.append((segs[2], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs[:15], start_i=4, refs_i=[0, 1]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=0))

        if i % 500 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_onu(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"Legenda:|" r"Foto:|" r"Acesse também o|" r"\[embed\]|" r"\[caption" r"",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile("(?:Foto|Imagem):.{1,80}$", re.MULTILINE)

    reg_full_document_skip = re.compile(
        r"Boletim semanal da ONU Brasil|" r"Visualize o boletim também em|" r"Confira o boletim da ONU Brasil" r"",
    )

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/onu_brasil/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_onu")):
        with open(uri, "r") as f_in:
            segs = [reg_noise_patterns.sub("", item.strip()) for item in f_in.read().strip().split("\n")]
            segs = [item for item in segs if item.strip() and not reg_banned_patterns.match(item)]

        if reg_full_document_skip.match(segs[9]):
            continue

        if long_segments:
            if len(segs) >= 13:
                seg_a = "\n".join(segs[9:13])
                seg_b = "\n".join(segs[13:])
                if min(len(seg_a), len(seg_b)) >= 64:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 12 and min(len(segs[9]), len(segs[11])) >= 48:
                pairs.append((segs[9], segs[11]))

            if len(segs) >= 13 and min(len(segs[12]), len(segs[11])) >= 64:
                pairs.append((segs[12], segs[11]))

            if len(segs) >= 14 and min(len(segs[12]), len(segs[13])) >= 64:
                pairs.append((segs[12], segs[13]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=14, refs_i=[9, 11]))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=13, context_i=9))

        if i % 2000 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs


def make_pairs_capes(*, long_segments: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    reg_banned_patterns = re.compile(
        r"Compartilhe|"
        r"link para copiar para a área de transferência|"
        r"link para Copiar para área de transferência|"
        r"Publicado em",
        re.IGNORECASE,
    )
    reg_noise_patterns = re.compile(r"\{mosimage\}")
    reg_fix_quote_start = re.compile(r"(?<=[^a-zçáéíóúàâêôãẽõü\s])\s*\?", re.IGNORECASE)
    reg_fix_quote_end = re.compile(r"\?\s*(?=[^a-zçáéíóúàâêôãẽõü\s])", re.IGNORECASE)
    reg_fix_comma = re.compile(r",(?=[^\s])")

    source_dir = os.path.join(utils.TESEMO_PATH, "outros/o1_noticias_governamentais/capes/*.txt")
    for i, uri in enumerate(tqdm.tqdm(glob.glob(source_dir), desc="news_capes")):
        with open(uri, "r") as f_in:
            segs = [reg_noise_patterns.sub("", item.strip()) for item in f_in.read().strip().split("\n")]
            segs = [
                reg_fix_comma.sub(
                    ", ",
                    reg_fix_quote_end.sub('" ', reg_fix_quote_start.sub(' "', item)),
                )
                for item in segs
                if len(item) >= 32 and not reg_banned_patterns.match(item)
            ]

        if len(segs) > 1 and segs[0].lower() == segs[1].lower():
            segs.pop(0)

        if long_segments:
            if len(segs) >= 4:
                seg_a = "\n".join(segs[:3])
                seg_b = "\n".join(segs[3:])
                if min(len(seg_a), len(seg_b)) >= 64:
                    pairs.append((seg_a, seg_b))

        else:
            if len(segs) >= 2 and min(len(segs[0]), len(segs[1])) >= 21:
                pairs.append((segs[0], segs[1]))

            if len(segs) >= 3 and min(len(segs[2]), len(segs[1])) >= 32:
                pairs.append((segs[2], segs[1]))

            if len(segs) >= 4 and min(len(segs[2]), len(segs[3])) >= 64:
                pairs.append((segs[2], segs[3]))

            pairs.extend(utils.fetch_laws_in_segments(segs=segs, start_i=4, refs_i=None))
            pairs.extend(utils.fetch_questions_in_segments(segs, start_i=3, context_i=0))

        if i % 200 == 0:
            utils.print_example(*pairs[-1])

    assert len(pairs)
    return pairs
