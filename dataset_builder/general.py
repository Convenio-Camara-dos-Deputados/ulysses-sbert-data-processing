import typing as t
import glob
import os
import collections
import hashlib

import regex as re
import pandas as pd
import numpy as np
import tqdm

from . import utils


_VISITED: dict[str, set[str]] = {
    "uri": set(),
    "source_name": set(),
}


def _do_intersect(i: int | slice, j: int | slice) -> bool:
    if not isinstance(i, slice):
        i = slice(i, i + 1)
    if not isinstance(j, slice):
        j = slice(j, j + 1)
    si, ei = i.start, (i.stop - 1 if i.stop else float("+inf"))
    sj, ej = j.start, (j.stop - 1 if j.stop else float("+inf"))
    return max(si, sj) <= min(ei, ej)


def _read_file_segments(
    uri: str,
    /,
    *,
    fn_text_preproc: t.Callable[[str], str] | None = None,
    min_seg_len: int = 1,
    redundancy_check_inds: tuple[int, int] | None = None,
    reg_banned_patterns: re.Pattern | None = None,
    fn_seg_preproc: t.Callable[[str], str] | None = None,
    fn_seg_postproc: t.Callable[[list[str]], list[str]] | None = None,
    apply_preproc_before_banned_patterns: bool = False,
    full_search_banned_patterns: bool = False,
) -> list[str]:
    with open(uri, "r", encoding="utf-8") as f_in:
        if not fn_text_preproc:
            segs = f_in.readlines()
        else:
            segs = fn_text_preproc(f_in.read()).split("\n")

    segs = [item.strip() for item in segs]
    segs = [item for item in segs if len(item) >= min_seg_len]

    if apply_preproc_before_banned_patterns and fn_seg_preproc:
        segs = [fn_seg_preproc(item) for item in segs]
        segs = [item for item in segs if len(item)]

    if reg_banned_patterns:
        fn_filter = reg_banned_patterns.search if full_search_banned_patterns else reg_banned_patterns.match
        segs = [item for item in segs if not fn_filter(item)]

    if not apply_preproc_before_banned_patterns and fn_seg_preproc:
        segs = [fn_seg_preproc(item) for item in segs]
        segs = [item for item in segs if len(item)]

    if redundancy_check_inds:
        i, j = redundancy_check_inds
        if len(segs) > max(i, j) and len(segs[i]) == len(segs[j]) and segs[i].lower() == segs[j].lower():
            segs.pop(i)

    if fn_seg_postproc:
        segs = fn_seg_postproc(segs)

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
    fetch_law_in_segments_kwargs: dict[str, t.Any] | None = None,
    fetch_questions_in_segments_kwargs: dict[str, t.Any] | None = None,
    fn_text_preproc: t.Callable[[str], str] | None = None,
    reg_filter_uris: re.Pattern | None = None,
    min_seg_len: int = 1,
    redundancy_check_inds: tuple[int, int] | None = None,
    reg_banned_patterns: re.Pattern | None = None,
    full_search_banned_patterns: bool = False,
    reg_document_full_skip: re.Pattern | None = None,
    document_full_skip_inds: int | list[int] | None = None,
    fn_seg_preproc: t.Callable[[str], str] | None = None,
    fn_seg_postproc: t.Callable[[list[str]], list[str]] | None = None,
    apply_preproc_before_banned_patterns: bool = False,
    long_segment_join_string: str = "\n",
    it_to_print: float | int | None = None,
    save_externally_for_evaluation: int | None = None,
) -> list[tuple[str, str]]:
    if uri in _VISITED["uri"]:
        raise ValueError(f"'{uri = }' has already been visited.")

    if source_name in _VISITED["source_name"]:
        raise ValueError(f"'{source_name = }' has already been visited.")

    if not reg_document_full_skip and document_full_skip_inds is not None:
        raise ValueError("'document_full_skip_inds' must be None when 'reg_document_full_skip=False'.")

    if fetch_law_in_segments and not fetch_law_in_segments_kwargs:
        raise ValueError('"fetch_law_in_segments_kwargs" must be provided when "fetch_law_in_segments=True".')

    if not fetch_law_in_segments and fetch_law_in_segments_kwargs is not None:
        raise ValueError("'fetch_law_in_segments_kwargs' must be None when 'fetch_law_in_segments=False'.")

    if fetch_questions_in_segments and not fetch_questions_in_segments_kwargs:
        raise ValueError('"fetch_questions_in_segments_kwargs" must be provided ' 'when "fetch_questions_in_segments=True".')

    if not fetch_questions_in_segments and fetch_questions_in_segments_kwargs is not None:
        raise ValueError("'fetch_questions_in_segments_kwargs' must be None when " "'fetch_questions_in_segments=False'.")

    if save_externally_for_evaluation and save_externally_for_evaluation <= 0:
        raise ValueError(f"'save_externally_for_evaluation' must be > 0.")

    _VISITED["uri"].add(uri)
    _VISITED["source_name"].add(source_name)

    pairs: list[tuple[str, str]] = []
    benchmark_pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(utils.Config.TESEMO_PATH, uri, "*.txt")

    uris = glob.glob(source_dir, recursive=False)

    if reg_filter_uris:
        uris = [uri for uri in uris if not reg_filter_uris.search(os.path.basename(uri))]

    if it_to_print is not None and 0.0 < it_to_print < 1.0:
        it_to_print = int(round(it_to_print * len(uris)))

    n: float | int = it_to_print or float("+inf")

    inds_for_benchmark: t.FrozenSet[int] = frozenset()

    if save_externally_for_evaluation:
        save_externally_for_evaluation = int(save_externally_for_evaluation)
        random_seed = int(hashlib.md5("dkjfkdsf".encode()).hexdigest(), base=16) % 2**32
        rng = np.random.RandomState(random_seed)
        inds_for_benchmark = rng.choice(len(uris), size=save_externally_for_evaluation, replace=False)
        inds_for_benchmark = frozenset(inds_for_benchmark)

    for k, file_uri in enumerate(tqdm.tqdm(uris, desc=source_name)):
        n -= 1

        cur_pair_container = pairs if k not in inds_for_benchmark else benchmark_pairs

        segs = _read_file_segments(
            file_uri,
            fn_text_preproc=fn_text_preproc,
            min_seg_len=min_seg_len,
            reg_banned_patterns=reg_banned_patterns,
            full_search_banned_patterns=full_search_banned_patterns,
            redundancy_check_inds=redundancy_check_inds,
            fn_seg_preproc=fn_seg_preproc,
            apply_preproc_before_banned_patterns=apply_preproc_before_banned_patterns,
            fn_seg_postproc=fn_seg_postproc,
        )

        skip_document = False

        if reg_document_full_skip:
            if isinstance(document_full_skip_inds, int):
                document_full_skip_inds = [document_full_skip_inds]

            for i in document_full_skip_inds:  # type: ignore
                if len(segs) > i and reg_document_full_skip.search(segs[i]):
                    skip_document = True
                    break

        if skip_document:
            continue

        if long_segments:
            i, j, min_len = long_segment_inds

            if _do_intersect(i, j):
                raise ValueError(
                    f"Intersection has been found for '{i}' and '{j}' " f"while building long segments for '{source_name=}'."
                )

            max_ind = max(
                int(i.start if isinstance(i, slice) else i),
                int(j.start if isinstance(j, slice) else j),
            )

            if len(segs) > max_ind:
                seg_a = long_segment_join_string.join(segs[i]) if isinstance(i, slice) else segs[i]
                seg_b = long_segment_join_string.join(segs[j]) if isinstance(j, slice) else segs[j]

                if min(len(seg_a), len(seg_b)) >= min_len:
                    cur_pair_container.append((seg_a, seg_b))

        else:
            for i, j, min_len in short_segment_inds:
                if len(segs) > max(i, j) and i != j:
                    seg_a, seg_b = segs[i], segs[j]
                    if min(len(seg_a), len(seg_b)) >= min_len:
                        cur_pair_container.append((seg_a, seg_b))

            if fetch_law_in_segments:
                cur_pair_container.extend(
                    utils.fetch_laws_in_segments(
                        segs,
                        **fetch_law_in_segments_kwargs,  # type: ignore
                    )
                )

            if fetch_questions_in_segments:
                cur_pair_container.extend(
                    utils.fetch_questions_in_segments(
                        segs,
                        **fetch_questions_in_segments_kwargs,  # type: ignore
                    )
                )

        if it_to_print and n <= 0 and cur_pair_container:
            n = it_to_print
            utils.print_example(*cur_pair_container[-1])

    if not pairs:
        raise ValueError(f"No pair has been generated ({source_name=}, {uri=}).")

    if save_externally_for_evaluation and not benchmark_pairs:
        raise ValueError(f"No pair has been generated ({source_name=}, {uri=}).")

    if len(pairs) <= len(benchmark_pairs):
        raise ValueError("Generated more benchmark pairs than regular pairs.")

    if benchmark_pairs:
        benchmark_df = utils.gen_dataframe(
            pairs=benchmark_pairs,
            source_name=source_name,
            max_length=10000,
        )
        utils.save_cache(
            [benchmark_df],
            output_dir="./benchmark_dfs",
            output_urn=f"benchmark_{source_name}.csv",
            sep=",",
        )
        print(f"Allocated {len(benchmark_df)} instances from '{source_name}' for benchmarking.")

    return pairs


def make_pairs_ministerios(*, long_segments: bool) -> dict[str, list[tuple[str, str]]]:
    pairs = collections.defaultdict(list)

    reg_foto = re.compile(r"[^\.a-zç]*\s*Fotos?:.+$", re.IGNORECASE)
    base_dir = os.path.join(utils.Config.TESEMO_PATH, "outros/o1_noticias_governamentais/ministerios")

    for uri in tqdm.tqdm(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)):
        with open(uri, "r", encoding="utf-8") as f_in:
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


def make_pairs_tv_camara(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_tv_camara"

    reg_banned_patterns = re.compile(r"Créditos?")

    reg_document_full_skip = re.compile(
        r"Revista da Câmara"
        r"|Veja o que"
        r"|Câmara Hoje"
        r"|agenda da Câmara"
        r"|Veja os destaques"
        r"|desta semana"
        r"|nesta semana"
        r"|semana passada"
        r"|na Câmara hoje"
        r"|Principais notícias"
        "",
        re.IGNORECASE,
    )

    pairs = _make_pairs_generic(
        "outros/o8_tv_camara",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(1, slice(2, None), 60),
        short_segment_inds=[
            (1, 2, 60),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=False,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [1, 2]},
        fetch_questions_in_segments_kwargs=None,
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=[1, 2],
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_radio_e_tv_justica(
    **kwargs,
) -> tuple[list[tuple[str, str]], str]:  # pylint: disable='unused-argument'
    df = pd.read_csv(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "radio_e_tv_justica_ementas.tsv"),
        sep="\t",
        index_col=0,
    )
    pairs = df.values.tolist()

    if not pairs:
        raise ValueError(f"No pair has been generated for '{source_name=}'.")

    return pairs, "radio_e_tv_justica"


def make_pairs_mpt(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_mpt"

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/mpt_ministerio_publico_do_trabalho",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 2, 40),
            (0, 3, 40),
            (4, 3, 40),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": [0, 2]},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_mpm(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_mpm"

    re_banned_patterns = re.compile(r"voltar\.+")

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/mpm_ministerio_publico_militar",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(1, 3), slice(3, None), 40),
        short_segment_inds=[
            (1, 2, 40),
            (3, 2, 40),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": [1, 2]},
        fetch_questions_in_segments_kwargs={"start_i": 3, "context_i": 1},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=re_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tcu(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_tcu"

    reg_document_full_skip = re.compile(r"Destaques da sessão plenária")

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/tcu_tribunal_de_contas_da_uniao",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 64),
            (0, 2, 64),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": [0, 1, 2]},
        fetch_questions_in_segments_kwargs={"start_i": 3, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=0,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_radio_camara(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "radio_camara"

    reg_document_full_skip = re.compile(
        r"(?:"
        r"loc[-–:]|"
        r"boa noite|"
        r"est[aá] no ar|"
        r"Edição [^0-9a-zç]|"
        r"(?:primeiro|segundo) bloco|"
        r"bloco [0-9]|"
        r"[aáà] seguir"
        r")",
        re.IGNORECASE,
    )

    def fn_seg_postproc(segs: list[str]) -> list[str]:
        out: list[str] = []
        for seg in segs:
            out.extend(utils.natural_sentence_tokenize(seg))
        return out

    pairs = _make_pairs_generic(
        "outros/o3_radio_camara",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 100),
        short_segment_inds=[
            (0, 1, 100),
        ],
        fetch_law_in_segments=False,
        fetch_questions_in_segments=False,
        fetch_law_in_segments_kwargs=None,
        fetch_questions_in_segments_kwargs=None,
        min_seg_len=100,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=[0, 1, 2] if long_segments else [0, 1],
        fn_seg_preproc=None,
        fn_seg_postproc=fn_seg_postproc,
        apply_preproc_before_banned_patterns=False,
        long_segment_join_string=" ",
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_trf4(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf4"

    reg_document_full_skip = re.compile("Agenda d[oa] presidente do TRF4", re.IGNORECASE)

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/trf4_tribunal_regional_federal_da_4_regiao",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(1, slice(4, None), 64),
        short_segment_inds=[
            (1, 4, 64),
            (5, 4, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 6, "refs_i": [1, 4]},
        fetch_questions_in_segments_kwargs={"start_i": 5, "context_i": 1},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=0,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_trf3(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf3"

    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.Config.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf3_tribunal_regional_federal_da_3_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc=source_name):
        with open(uri, "r", encoding="utf-8") as f_in:
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

    if not pairs:
        raise ValueError(f"No pair has been generated for '{source_name=}'.")

    return pairs, source_name


def make_pairs_trf2(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf2"

    pairs: list[tuple[str, str]] = []

    source_dir = os.path.join(
        utils.Config.TESEMO_PATH,
        "outros/o1_noticias_governamentais/trf2_tribunal_regional_federal_da_2_regiao/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc=source_name):
        with open(uri, "r", encoding="utf-8") as f_in:
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

    if not pairs:
        raise ValueError(f"No pair has been generated for '{source_name=}'.")

    return pairs, source_name


def make_pairs_trf1(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf1"

    pairs: list[tuple[str, str]] = []

    reg_split_1 = re.compile(
        r"(?<![^\s])(?:\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2})(?![^\s])",
        re.IGNORECASE,
    )

    reg_split_2 = re.compile(
        r"(?<![^\s])Cr[eé]ditos?:\s+(?:imagem da web|Google imagens|Reprodução/Internet|Internet)",
        re.IGNORECASE,
    )

    def fn_text_preproc(x: str) -> str:
        x = reg_split_1.sub("\n", x, count=1)
        x = reg_split_2.sub("\n", x, count=1)
        return x

    reg_noise_patterns = re.compile(r"(?:DECISÃO|INSTITUCIONAL):\s*")

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    reg_banned_patterns = re.compile(r"Créditos?:") if not long_segments else None

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/trf1_tribunal_regional_federal_da_1_regiao",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=fn_text_preproc,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_stj(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_stj"

    reg_banned_patterns = re.compile(r"Atualizad[oa] em|^[0-9]{2}/[0-9]{2}")

    def fn_seg_postproc(segs: list[str]) -> list[str]:
        segs = segs[1:]
        segs = utils.natural_sentence_tokenize(" ".join(segs))
        return segs

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/stj_superior_tribunal_de_justica",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=5,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=fn_seg_postproc,
        apply_preproc_before_banned_patterns=False,
        long_segment_join_string=" ",
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tse(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_tse"

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/tse_tribunal_superior_eleitoral",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(3, None), 64),
        short_segment_inds=[
            (0, 3, 48),
            (1, 3, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": [0, 3]},
        fetch_questions_in_segments_kwargs={"start_i": 3, "context_i": 0},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_cnmp(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_cnmp"

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/cnmp_conselho_nacional_do_ministerio_publico",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(2, 4), slice(4, None), 48),
        short_segment_inds=[
            (2, 3, 48),
            (4, 3, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": [2, 3]},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 2},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_bc(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_bc"

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/banco_central",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 48),
        short_segment_inds=[
            (0, 1, 64),
            (0, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_camara_comments(
    **kwargs,
) -> tuple[list[tuple[str, str]], str]:  # pylint: disable='unused-argument'
    source_name = "news_camara_comments"

    pairs: list[tuple[str, str]] = []

    reg = re.compile(r"https_www_camara_leg_br_noticias_\d+_")
    source_dir = os.path.join(
        utils.Config.TESEMO_PATH,
        "outros/o1_noticias_governamentais/comentarios_noticias_camara_dos_deputados/*.txt",
    )
    for uri in tqdm.tqdm(glob.glob(source_dir), desc=source_name):
        with open(uri, "r", encoding="utf-8") as f_in:
            segs = f_in.read().strip().split("\n")

        split_1 = reg.sub("", os.path.basename(uri)).replace(".txt", "").replace("_", " ")

        for split_2 in segs:
            if len(split_2) >= 30:
                pairs.append((split_1, split_2))

    if not pairs:
        raise ValueError(f"No pair has been generated for '{source_name=}'.")

    return pairs, source_name


def make_pairs_camara(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_camara"

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/camara_dos_deputados",
        source_name=source_name,
        reg_filter_uris=None,
        fn_text_preproc=None,
        long_segments=long_segments,
        long_segment_inds=(1, slice(2, None), 48),
        short_segment_inds=[
            (1, 2, 48),
            (1, 3, 48),
            (4, 3, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 1},
        min_seg_len=25,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_senado(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_senado"

    reg_split = re.compile(
        r"^\s*(?:Saiba mais|Buscar|Mais vídeos|MAIS NOTÍCIAS SOBRE|Mais informações a seguir|Mais vistas|Últimas:)$",
        re.IGNORECASE | re.MULTILINE,
    )
    reg_banned_patterns = re.compile(
        r"^Compartilhe este conteúdo no"
        r"|^Home"
        + (r"|^Agência Senado \(Reprodução autorizada mediante citação da Agência Senado\)" if not long_segments else r"")
        + r"|^Compartilhar:"
        r"|^Compartilhe:"
        r"|[0-9]{2}/[0-9]{2}/[0-9]{4}\s*,\s+[0-9]{2}[:h][0-9]{2}"
        r"|^O projeto foi relatado.{,60}$"
        "",
        re.IGNORECASE,
    )

    reg_filter_uris = re.compile(r"cpi", re.IGNORECASE)

    def fn_text_preproc(x: str) -> str:
        x = reg_split.split(x)[0]
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/senado_federal",
        source_name=source_name,
        reg_filter_uris=reg_filter_uris,
        fn_text_preproc=fn_text_preproc,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=30,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_stf(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_stf"

    reg_banned_patterns = re.compile(
        r"^.{,80}atualizado há"
        r"|pessoas já viram isso"
        r"|Pauta de julgamentos previstos"
        r"|Confira, abaixo, o resumo dos julgamentos previstos"
        r"",
        re.IGNORECASE,
    )
    reg_document_full_skip = re.compile(
        r"Agenda d[ao] presidente"
        r"|Confira os destaques da TV Justiça"
        r"|Destaques da Rádio Justiça"
        r"|Confira a programação"
        r"",
        re.IGNORECASE,
    )

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/stf_superior_tribunal_federal",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=0,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_stm(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_stm"

    reg_banned_patterns = re.compile(
        r"\d{2}/\d{2}/\d{4}$"
        r"|Imprimir\s*E-mail\s*"
        r"|Imagem ilustrativa"
        r"|foto: .{,80}$"
        r"|Conselho Nacional de Justiça$"
        r"|Supremo Tribunal Federal$"
        r"|Crédito: .{,80}$"
        r"",
        re.IGNORECASE,
    )

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/stm_superior_tribunal_militar",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tst(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_tst"

    reg_banned_patterns = re.compile(
        r"Esse conteúdo não está disponível sem cookies"
        r"|null$"
        r"|imprimir$"
        r"|Notícias do TST$"
        r"|Seguir$"
        r"|\([STQD]"
        r"",
        re.IGNORECASE,
    )

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/tst_tribunal_superior_do_trabalho",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tst_radio(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "tst_radio"

    reg_document_full_skip = re.compile(r"Trabalho e Justiça", re.IGNORECASE)

    reg_banned_patterns = re.compile(
        r"^null$"
        r"|^\d{2}.\d{2}.\d{4}\s*.{,5}$"
        r"|Baixe o [aá]udio"
        r"|Reproduzir o áudio"
        r"|O programa Trabalho e Justiça vai ao ar"
        r"|Ouça os detalhes na reportagem"
        r"|Permitida a reprodução mediante citação da fonte"
        r"|Esta matéria tem caráter informativo, sem cunho oficial"
        r"|Esse conteúdo não está disponível sem cookies"
        r"|E também acompanha"
        r"|E mais:"
        r"|Você também vai saber"
        r"|Você também vai var"
        r"|Rádio Justiça"
        r"|^Processo:"
        r"|^Mais detalhes$"
        r"|^Para saber mais, aperte o play!"
        r"|\([STQD]"
        r"",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile(r"^REPÓRTER:\s*|\s*\|\s*TST na Voz do Brasil", re.IGNORECASE)

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o7_radio_tst",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(0, slice(1, None), 32),
        short_segment_inds=[
            (0, 1, 32),
        ],
        fetch_law_in_segments=False,
        fetch_questions_in_segments=False,
        fetch_law_in_segments_kwargs=None,
        fetch_questions_in_segments_kwargs=None,
        min_seg_len=1,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=0,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tst_tv(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "tst_tv"

    reg_banned_patterns = re.compile(
        r"^null$"
        r"|^\s*Esse conteúdo não está disponível sem cookies"
        r"|E também acompanha"
        r"|Você também vai saber"
        r"|você também vai ver"
        r"|^\s*\(.{,45}\d{2,4}\)\s*$"
        r"|^Mais detalhes$"
        r"|^Processo:"
        r"|^\d{2}/\d{2}/\d{2,4}\s*.?$",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile(r"\| Programa completo", re.IGNORECASE)

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    reg_document_full_skip = re.compile(
        r"Nesta edição você confere"
        r"|Confira nessa edição"
        r"|Veja os destaques desta edição"
        r"|Nesta edição você confere"
        r"|Veja os destaques desta edição"
        r"",
        re.IGNORECASE,
    )

    pairs = _make_pairs_generic(
        "outros/o10_tst_tv",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(0, slice(1, None), 21),
        short_segment_inds=[
            (0, 1, 21),
        ],
        fetch_law_in_segments=False,
        fetch_questions_in_segments=False,
        fetch_law_in_segments_kwargs=None,
        fetch_questions_in_segments_kwargs=None,
        min_seg_len=20,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=0,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_trf6(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf6"

    reg_noise_patterns = re.compile(r"Início\s*»\s*", re.IGNORECASE)

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/trf6_tribunal_regional_federal_da_6_regiao",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 48),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 3, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 2, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=False,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_trf5(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_trf5"

    reg_noise_patterns = re.compile(r"Início\s*»\s*", re.IGNORECASE)
    reg_banned_patterns = re.compile("Última atualização:|^Por:", re.IGNORECASE)

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    def fn_seg_postproc(segs: list[str]) -> list[str]:
        out: list[str] = []
        for seg in segs:
            out.extend(utils.natural_sentence_tokenize(seg))
        return out

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/trf5_tribunal_regional_federal_da_5_regiao",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 160),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": [0, 1]},
        fetch_questions_in_segments_kwargs={"start_i": 3, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=fn_seg_postproc,
        apply_preproc_before_banned_patterns=False,
        long_segment_join_string=" ",
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_onu(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_onu"

    reg_banned_patterns = re.compile(
        r"Legenda:|Fotos?:|Acesse também o|\[embed\]|\[caption",
        re.IGNORECASE,
    )

    reg_noise_patterns = re.compile("(?:Foto|Imagem):.{1,80}$", re.MULTILINE)

    reg_document_full_skip = re.compile(
        r"^Boletim semanal da ONU Brasil|^Visualize o boletim também em|^Confira o boletim da ONU Brasil",
    )

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/onu_brasil",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(9, 13), slice(13, None), 64),
        short_segment_inds=[
            (9, 11, 48),
            (12, 11, 64),
            (12, 13, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 14, "refs_i": [9, 11]},
        fetch_questions_in_segments_kwargs={"start_i": 13, "context_i": 9},
        min_seg_len=0,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=9,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_capes(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_capes"

    reg_banned_patterns = re.compile(
        r"Compartilhe"
        r"|link para copiar para a área de transferência"
        r"|link para Copiar para área de transferência"
        r"|Publicado em",
        re.IGNORECASE,
    )
    reg_noise_patterns = re.compile(r"\{mosimage\}")
    reg_fix_quote_start = re.compile(r"(?<=[^a-zçáéíóúàâêôãẽõü\s])\s*\?", re.IGNORECASE)
    reg_fix_quote_end = re.compile(r"\?\s*(?=[^a-zçáéíóúàâêôãẽõü\s])", re.IGNORECASE)
    reg_fix_comma = re.compile(r",(?=[^\s])")

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise_patterns.sub("", x)
        x = reg_fix_quote_start.sub(' "', x)
        x = reg_fix_quote_end.sub('" ', x)
        x = reg_fix_comma.sub(", ", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/capes",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 21),
            (2, 1, 32),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 3, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=(0, 1),
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_tjac(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ac"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4} \| Notícias")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 2, 32),
            (0, 3, 48),
            (4, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 5, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjal(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "al"
    source_name = f"news_tj{state}"

    reg_imagem = re.compile(r"(?:Foto|Imagem|Arte):.{,60}$", re.IGNORECASE)

    def fn_seg_preproc(x: str) -> str:
        x = reg_imagem.sub("", x)
        x = x.strip()
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjam(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "am"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjap(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ap"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjba(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ba"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("PUBLICADO EM:" "|ATUALIZADO EM:" "|Texto:" "|COMPARTILHAR:" "|IMPRIMIR$")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjce(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ce"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Ouvir:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjdf(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "df"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("©|Todos os direitos reservados|Publique essa notícia no")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjes(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "es"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("(?:SEGUNDA|TERÇA|QUARTA|QUINTA|SEXTA)-FEIRA,")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (0, 2, 48),
            (3, 2, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjgo(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "go"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (1, 2, 32),
            (3, 2, 48),
            (3, 4, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 5, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjma(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ma"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("foto/divulgação:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjmg(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "mg"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("[0-9]{1,2}/[0-9]{1,2}/[0-9]{4} [0-9]{1,2}h" "|Número de visualizações")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjms(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ms"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("[0-9]{1,2}/[0-9]{1,2}/[0-9]{4} [0-9]{1,2}h")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjmt(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "mt"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjpa(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pa"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjpb(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pb"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Compartilhar: ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjpe(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pe"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjpi(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pi"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile(
        "Publicado por:" "|(?:segunda|terça|quarta|quinta|sexta)-feira, " "|(?:domingo|s[aá]bado), [0-9]{1,2}"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjpr(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pr"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 2, 32),
            (3, 2, 48),
            (3, 4, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 5, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 5, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjrj(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rj"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Notícia publicada por")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjrn(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rn"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Publicado em:" "|PORTAL ANTIGO")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjro(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ro"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile(
        "(Segunda|Terça|Quarta|Quinta|Sexta|Sábado|Domingo),"
        "|Tamanho da Fonte"
        "|Tweet$"
        "|Voltar ao topo"
        "|Transmissão ao vivo -"
        "|TRIBUNAL DE JUSTIÇA DO ESTADO DE RONDÔNIA"
        "|COORDENADORIA DE COMUNICAÇÃO SOCIAL"
        "|Porto Velho, [0-9]{1,2}"
        "|Seção de Imprensa e Divulgação"
        "|JUser: "
        "|JavaScript is currently disabled"
        "|Assessoria de Comunicação Instituciona(?:is|l)"
        "|Clique aqui e assista a transmissão ao vivo\."
    )

    reg_document_full_skip = re.compile(
        "NOTÍCIAS LOCAIS, NACIONAIS E INTERNACIONAIS"
        "|NOTÍCIAS DO JUDICIÁRIO DE RONDÔNIA"
        "|Sessão da 1ª Câmara Criminal do TJRO"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=[0],
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjrr(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rr"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjrs(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rs"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Compartilhar:" "|Texto:" "|Créditos:" "|[0-9]{1,2}/[0-9]{1,2}/[0-9]{4} - ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjsc(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "sc"
    source_name = f"news_tj{state}"

    reg_banned_patterns = re.compile("Copiar o link desta notícia\." "|Imagens:" "|Conteúdo:" "|Responsável:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjse(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "se"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjsp(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "sp"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_tjto(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "to"
    source_name = f"news_tj{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/tribunais_de_justica/tj{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ac(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ac"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "^Da Redação "
        "|(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro) [0-9]{1,2}, [0-9]{4} - [0-9]{1,2}h[0-9]{1,2}min"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=True,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_al(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "al"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("[0-9]{1,2}/[0-9]{1,2}/[0-9]{4} [0-9]{1,2}:[0-9]{1,2} \|")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_am(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "am"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Por .{60}$")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ap(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ap"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "(segunda|terça|quarta|quinta|sexta|sábado|domingo), [0-9]{1,2}" "|Tweet" "|Por: " "|Fotos?: "
    )

    def fn_text_preproc(x: str) -> str:
        x = x.split("NOTÍCIAS RELACIONADAS")[0]
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=fn_text_preproc,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ba(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ba"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "[0-9]{1,2} (?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro) [0-9]{4}\s*$"
        "|Início Notícias"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ce(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ce"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Data: " "|Categoria: ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_df(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "df"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4} [aà]s [0-9]{1,2}:[0-9]{1,2}" "|.{,70}, da Agência Brasília"
    )

    def fn_text_preproc(x: str) -> str:
        x = x.split("ÚLTIMAS NOTÍCIAS")[0]
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=fn_text_preproc,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_es(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "es"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4} [0-9]{1,2}[h:][0-9]{1,2}" "|Tweet" "|Linkedin" "|Compartilhar" "|Imprimir" "|Fotos?:"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_go(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "go"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Publicado:" "|Última Atualização:" "|Mais informações:")

    reg_document_full_skip = re.compile("Esses são os destaques d[oa]" "|Agenda d[ao] governadora?")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=[0],
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ma(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ma"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(" - ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_mg(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "mg"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("(?:SEG|TER|QUA|QUI|SEX|S[ÁA]B|DOM) [0-9]{1,2} ")

    def fn_text_preproc(x: str) -> str:
        x = x.split("OUTRAS NOTÍCIAS")[0]
        return x

    def fn_seg_preproc(x: str) -> str:
        x = x.replace("[RÁDIO]", "").strip()
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=fn_text_preproc,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=[0, 2],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ms(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ms"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(".{,60}nrodrigues" "|http")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_mt(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "mt"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Crédito - " "|http")

    def fn_seg_preproc(x: str) -> str:
        x = x.removesuffix(" - mt.gov.br")
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_pa(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pa"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Por Redação - ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_pb(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pb"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("publicado:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_pe(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pe"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("PUBLICADO:" "|Fotos?:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_pi(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pi"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("" "|Repórter(?:es)?:")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 3), slice(3, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_pr(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "pr"
    source_name = f"news_state_{state}"

    reg_noise = re.compile(r" -.{,40}$")

    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    def fn_seg_postproc(segs: list[str]) -> list[str]:
        i = 0
        while i < len(segs) - 1:
            if segs[i].lower().strip() == segs[i + 1].lower().strip():
                segs.pop(i)
            else:
                i += 1

        if len(segs) >= 2 and segs[-1].lower().strip() == segs[-2].lower().strip():
            segs.pop(-1)
            segs.pop(-2)

        return segs

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=[0, 2],
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=fn_seg_postproc,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_rj(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rj"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("Publicado [0-9]{1,2}/[0-9]{1,2}/[0-9]{4}" "|null ")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_rn(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rn"
    source_name = f"news_state_{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_ro(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "ro"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "[0-9]+.{,60} \| Governo do Estado de Rondônia$",
    )

    def fn_text_preproc(x: str) -> str:
        x = x.split("Leia mais")[0]
        return x

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=fn_text_preproc,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_rr(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rr"
    source_name = f"news_state_{state}"

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_rs(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "rs"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(r"Inicial Imprensa" r"|Publicação:" r"|.{,60}\.jpg\b")

    reg_document_full_skip = re.compile("Agenda d[ao] governadora?")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=reg_document_full_skip,
        document_full_skip_inds=[0],
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_sc(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "sc"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        r"Por.{,60}[0-9] de [a-zç]+ de [0-9]{4}$" "|Secretaria de Estado da Justiça e Cidadania – SJC"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_se(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "se"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("(?:Segunda|Terça|Quarta|Quinta|Sexta)-Feira," "|(?:Sábado|Domingo),")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 40),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=40,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_sp(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "sp"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile(
        "(:?seg|ter|qua|qui|sex|s[aá]b|dom), " "|Compartilhe essa página" "|[-–] Siga o Governo do Estado"
    )

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_state_to(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    state = "to"
    source_name = f"news_state_{state}"

    reg_banned_patterns = re.compile("publicado:" "|por .{,50}$")

    pairs = _make_pairs_generic(
        f"outros/o1_noticias_governamentais/governos_estaduais/{state}",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=None,
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
        save_externally_for_evaluation=100,
    )

    return pairs, source_name


def make_pairs_gsi(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_gsi"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/gsi_gabinete_de_seguranca_institucional",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_secom(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_secom"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/secom_secretaria_de_comunicacao_social",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_cgu(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_cgu"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/cgu_controladoria_geral_da_uniao",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_agu(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_agu"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/agu_advocacia_geral_da_uniao",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_casa_civil(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_casa_civil"

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/casa_civil",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=None,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_ancine(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_ancine"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/ancine_agencia_nacional_do_cinema",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_funai(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_funai"

    reg_banned_patterns = re.compile(
        "Compartilhe:"
        "|Compartilhe por Facebook"
        "|Compartilhe por Twitter"
        "|Compartilhe por WhatsApp"
        "|link para Copiar para área de transferência"
        "|Publicado em"
    )

    reg_noise = re.compile(r" - Foto:.{,80}$")
    def fn_seg_preproc(x: str) -> str:
        x = reg_noise.sub("", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/funai",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 48),
            (2, 1, 64),
            (2, 3, 80),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=48,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=fn_seg_preproc,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_bnds(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_bnds"

    reg_banned_patterns = re.compile(
        r"[0-9]+ de "
        r"(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro) "
        r"de [0-9]{4}"
    )

    reg_subitems = re.compile(r"\n\s*[•·.]\s+")
    def fn_text_preproc(x: str) -> str:
        x = reg_subitems.sub("; ", x)
        return x

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/bnds_banco_nacional_do_desenvolvimento",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=fn_text_preproc,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name


def make_pairs_fapesp(*, long_segments: bool) -> tuple[list[tuple[str, str]], str]:
    source_name = "news_fapesp"

    reg_banned_patterns = re.compile(
        "[0-9]+ de (?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro) de [0-9]{4}"
        "|Facebook"
        "|Twitter"
        "|Email"
        "|WhatsApp"
        "|LinkedIn"
        "|VOLTAR"
        "|\s*Republicar"
    )

    pairs = _make_pairs_generic(
        "outros/o1_noticias_governamentais/fapesp",
        source_name=source_name,
        long_segments=long_segments,
        long_segment_inds=(slice(0, 2), slice(2, None), 64),
        short_segment_inds=[
            (0, 1, 32),
            (2, 1, 48),
            (2, 3, 64),
        ],
        fn_text_preproc=None,
        fetch_law_in_segments=True,
        fetch_questions_in_segments=True,
        fetch_law_in_segments_kwargs={"start_i": 4, "refs_i": None},
        fetch_questions_in_segments_kwargs={"start_i": 4, "context_i": 0},
        min_seg_len=32,
        redundancy_check_inds=[0, 1],
        reg_banned_patterns=reg_banned_patterns,
        full_search_banned_patterns=False,
        reg_document_full_skip=None,
        document_full_skip_inds=None,
        fn_seg_preproc=None,
        fn_seg_postproc=None,
        apply_preproc_before_banned_patterns=True,
        it_to_print=utils.Config.IT_TO_PRINT,
    )

    return pairs, source_name
