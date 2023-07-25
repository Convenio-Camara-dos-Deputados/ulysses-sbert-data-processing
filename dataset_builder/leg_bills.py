import typing as t
import collections
import os
import glob

import regex as re
import pandas as pd
import tqdm
import segmentador

from . import utils


reg_accepted_documents_name = re.compile("|".join(["PROJETO", "LEI", "DECRETO"]), re.IGNORECASE)
reg_table = re.compile(r"QUADRO|TABELA|TOTAL")
reg_nova_ementa = re.compile(r"\s*nova\s*ementa\s*.{,70}:", re.IGNORECASE)
reg_sala = re.compile("\s*Sala\s+d[ae]s?\s+(?:Sess|Comiss)(?:[aã]o|[oõ]es)", re.IGNORECASE)
reg_someone = re.compile(
    r"(?:[OA]\s+(?:VICE[-\s])?(?:PRESIDENT[AE]|GOVERNADORA?|MESA DA ASSEMBL[ÉE]IA|PREFEIT[OA]|SECRET[AÁ]RI[OA]))|(?:Legislação\s+)?GOVERNO\s*D",
    re.IGNORECASE,
)

reg_preproc_ba_art = re.compile(r"(?<=[^\s])Art\.")
reg_preproc_ba_law = re.compile(
    r"(R E S O L V E|D E C R E T A|Pareceres Tributários|PGE - BA|PRESIDENTE|GOVERNADOR|Presidente|Governador|Senador|Prefeito|Planejamento|Assembleia|Secretário|Civil|Anexo|Único|ANEXO|ÚNICO|(?:DECRETO|LEI)[-A-Z\sÇÁÉÍÓÚÀÂÊÔÃÕẼÜ]{,30} N[oº] [\s0-9\.]+ DE [0-9]+ DE \w+ DE [0-9]+)"
)
reg_preproc_ba_whitespaces = re.compile(r"\h{2,}")


def split_justificativa(
    content: str, segmenter: segmentador.Segmenter
) -> tuple[t.Optional[str], t.Optional[str]]:
    content = content.split("ANEXO")[0]
    just_content = segmenter.preprocess_legal_text(content, return_justificativa=True)[1]

    if not just_content:
        return None, None

    just_content = just_content[0]

    if reg_table.search(just_content[:500]):
        return None, None

    aux = "\n".join(segmenter(just_content, remove_noise_subsegments=True))
    if len(aux) > 0.40 * len(just_content):
        just_content = aux

    just_a, i = utils.fetch_further_leg_context(utils.natural_sentence_tokenize(just_content))
    just_b, _ = utils.fetch_further_leg_context(
        utils.natural_sentence_tokenize(just_content), start_i=i
    )

    if just_a and reg_sala.match(just_a):
        just_a = None
    if just_b and reg_sala.match(just_b):
        just_b = None

    return just_a or None, just_b or None


def make_pairs_fed_bills(
    debug: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    # # Older version:
    # documents = []
    # for uri in glob.glob(os.path.abspath("./raw_data/proposicao-tema-*.csv")):
    #     documents.append(pd.read_csv(uri, usecols=["txtNome", "txtEmenta", "imgArquivoTeorPDF"], index_col=0))
    # assert len(documents)
    # documents = pd.concat(documents)

    # New version:
    documents = pd.read_csv(
        os.path.abspath("./raw_data/USP_dataset_proposicoes_20230428.csv"),
        usecols=["txtNome", "txtEmenta", "txtExplicacaoEmenta", "txtInteiroTeor"],
        index_col=0,
    )

    assert len(documents)

    documents.dropna(subset=["txtEmenta", "txtInteiroTeor"], inplace=True)
    documents = documents.loc[~documents.index.duplicated(keep="last")]
    documents = documents.filter(
        regex="|".join(["PL", "PDC", "PEC", "PLP", "PRC", "PDL"]), axis="index"
    )

    assert len(documents)

    if debug:
        documents = documents.iloc[:50, :]

    pairs_leg: list[tuple[str, str]] = []
    pairs_just: list[tuple[str, str]] = []
    pbar = tqdm.tqdm(documents.iterrows(), desc="(federal leg bills)", total=len(documents))
    iters_to_print = max(int(len(pbar) * 0.05), 1)

    segmenter = segmentador.BERTSegmenter(device="cuda:0")

    min_length = 48

    for i, (doc_name, (ementa_content, ementa_explanation, doc_content)) in enumerate(pbar, 1):
        ementa_content = reg_nova_ementa.split(ementa_content)[-1]
        ementa_content = ementa_content.strip()
        ementa_content = ementa_content.replace("\n", " ")
        ementa_content = utils.remove_spurious_whitespaces(ementa_content)

        ementa_content_ext = f"({doc_name}) {ementa_content}"

        if isinstance(ementa_explanation, str) and len(ementa_explanation) >= min_length:
            ementa_explanation = ementa_explanation.strip()
            pairs_leg.append((ementa_content_ext, ementa_explanation))

        first_article = utils.fetch_first_item(doc_content, segmenter=segmenter, prefix="Art")
        first_article = utils.remove_spurious_whitespaces(first_article)

        if first_article and len(first_article) >= min_length:
            pairs_leg.append((ementa_content_ext, first_article))

        just_a, just_b = split_justificativa(doc_content, segmenter=segmenter)

        if just_a and min(len(ementa_content), len(just_a)) >= min_length:
            pairs_just.append((ementa_content, just_a))

        if just_b and just_a and min(len(just_b), len(just_a)) >= min_length:
            pairs_just.append((just_b, just_a))

        if i % iters_to_print == 0:
            if pairs_leg:
                utils.print_example(*pairs_leg[-1])
            if len(pairs_just) >= 2:
                utils.print_example(*pairs_just[-2])
            if len(pairs_just) >= 1:
                utils.print_example(*pairs_just[-1])

    assert len(pairs_leg)
    assert len(pairs_just)

    return pairs_leg, pairs_just


def fn_patch_return(fn):
    def gn(*args, **kwargs) -> tuple[str | None, str | None]:
        ret = fn(*args, **kwargs)
        return ret or (None, None)

    return gn


@fn_patch_return
def fn_doc_name_ac(segs: list[str], *args: t.Any, **kwargs: t.Any) -> tuple[str, str] | None:
    for i, seg in enumerate(segs):
        if reg_someone.match(seg):
            return segs[i - 2], segs[i - 1]


@fn_patch_return
def fn_doc_name_al(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    i = first_art_ind - 1
    reg_skip = re.compile(
        "Este texto não|Autora?:|[AO] GOVERNADORA? DO ESTADO|PUBLICAD[OA] NO DOE|O PRESIDENTE",
        re.IGNORECASE,
    )
    while i >= 1 and (reg_skip.match(segs[i]) or reg_skip.match(segs[i - 1])):
        i -= 1

    if i >= 1:
        return segs[i - 1], segs[i]


@fn_patch_return
def fn_doc_name_am(segs: list[str], *args: t.Any, **kwargs: t.Any) -> tuple[str, str] | None:
    for j, seg in enumerate(segs):
        if reg_someone.match(seg):
            break
    else:
        return None

    reg_skip = re.compile(
        "\s*[•·]|Prorrogad|Alterad|Revogad|[AO] GOVERNADORA?|(Re)?publicad[ao] no DOE|CONSIDERANDO",
        re.IGNORECASE,
    )
    i = j - 2
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], re.sub(
            ".*Publicad[ao] no DOE.{,45}p([áa]g)?\. [0-9]+\s*(\.\s*)?",
            "",
            segs[j - 1],
            flags=re.IGNORECASE,
        )


@fn_patch_return
def fn_doc_name_ap(segs: list[str], *args: t.Any, **kwargs: t.Any) -> tuple[str, str] | None:
    for j, seg in enumerate(segs):
        if reg_someone.match(seg):
            break
    else:
        return None

    reg_skip = re.compile("Autores:|Autora?s?:|Autoria:|Publicad[ao]", re.IGNORECASE)
    i = j - 2
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[j - 1]


@fn_patch_return
def fn_doc_name_es(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    i = first_art_ind - 3
    while i >= 0 and reg_someone.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], re.sub(
            "\s*[AO] GOVERNADORA? DO ESTADO DO ESP[IÍ]RITO SANTO\s*",
            "",
            segs[first_art_ind - 2],
            flags=re.IGNORECASE,
        )


@fn_patch_return
def fn_doc_name_go(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Este texto não|Mensagem de Veto|Revogad", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_ma(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("A MESA DA ASSEMBL[ÉE]IA|ESTADO DO MARANHÃO ASSEMBL[ÉE]IA", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_mg(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile(
        "Palácio da Inconfidência|Belo Horizonte, aos|Considerando|DECRETA", re.IGNORECASE
    )
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_ms(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Revogad|Publicad|[AO] GOVERNADOR|DECRETA", re.IGNORECASE)
    i = first_art_ind - 2
    while i >= 1 and (reg_skip.match(segs[i]) or reg_skip.match(segs[i - 1])):
        i -= 1

    if i >= 0:
        return segs[i - 1], segs[i]


@fn_patch_return
def fn_doc_name_mt(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Autoria:|Autores:|Autora?s?:", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_pa(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Este texto não|Mensagem de Veto|Revogad", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_pb(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Autoria:|Autores:|Autora?s?:", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_pe(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Este texto não|DECRETA|CONSIDERANDO", re.IGNORECASE)
    i = first_art_ind - 2
    while i >= 1 and (
        reg_skip.match(segs[i])
        or reg_someone.match(segs[i])
        or reg_skip.match(segs[i - 1])
        or reg_someone.match(segs[i - 1])
    ):
        i -= 1

    if i >= 1:
        return segs[i - 1], segs[i]


@fn_patch_return
def fn_doc_name_pr(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile("Publicad", re.IGNORECASE)
    i = first_art_ind - 3
    while i >= 0 and reg_skip.match(segs[i]):
        i -= 1

    if i >= 0:
        return segs[i], re.sub(r"Súmula:\s*", "", segs[first_art_ind - 2], flags=re.IGNORECASE)


@fn_patch_return
def fn_doc_name_sc(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile(
        r".{,45}:|Documentação|Alterad|Revogad|Consolidad|Veto|LIVRO|TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO|Procedência|Natureza|\bD[\s.]*O\b",
        re.IGNORECASE,
    )
    i = first_art_ind - 3
    while i >= 0 and (reg_skip.match(segs[i]) or len(segs[i]) <= 45):
        i -= 1

    if i >= 0:
        return segs[i], segs[first_art_ind - 2]


@fn_patch_return
def fn_doc_name_sp(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile(
        "(?:Re)?publicad[oa]|Revogad[oa]|Prorrogad[oa]|Alterad[oa]|Rejeitad[oa]|Consolidad[oa]|Regulamentad[oa]|"
        "Este texto não|"
        "CONSIDERANDO|"
        "Mensagem de Veto|"
        "DECRETA|"
        "RESOLVE|"
        "LIVRO|CAP[IÍ]TULO|SEÇÃO|SUBSEÇÃO|PARTE|T[ÍI]TULO|"
        "(?:Autoria|Autores|Autora?s?):"
        "",
        re.IGNORECASE,
    )

    i = first_art_ind - 2
    while i >= 1 and (
        min(len(segs[i]), len(segs[i - 1])) <= 5
        or reg_skip.match(segs[i])
        or reg_someone.match(segs[i])
        or reg_skip.match(segs[i - 1])
        or reg_someone.match(segs[i - 1])
    ):
        i -= 1

    if i >= 1:
        title = segs[i - 1]
        ementa = re.sub(
            "\s*(?:[A-ZÇÀÁÉÍÓÚÂÊÔÃẼÕÜ]\s*)+, GOVERNADOR DO ESTADO DE SÃO PAULO.*$", "", segs[i]
        )
        bad_suffix = ", dos Deputados"
        if title.endswith(bad_suffix):
            title = title[-len(bad_suffix) :]
            ementa = ementa.split(")")[-1]
        return title, ementa


@fn_patch_return
def fn_doc_name_to(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile(
        r"Publicad|Alterad|Regulamentad|Revogad|Consolidad|Veto|LIVRO|TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO|\s*\*",
        re.IGNORECASE,
    )
    i = first_art_ind - 2
    while i >= 1 and (reg_skip.match(segs[i]) or reg_someone.match(segs[i]) or len(segs[i]) <= 45):
        i -= 1

    j = i - 1
    while j >= 0 and (reg_skip.match(segs[j]) or reg_someone.match(segs[j]) or len(segs[j]) <= 45):
        j -= 1

    if i >= 1 and j >= 0:
        return segs[j], segs[i]


@fn_patch_return
def fn_doc_name_default(
    segs: list[str], first_art_ind: int, *args: t.Any, **kwargs: t.Any
) -> tuple[str, str] | None:
    reg_skip = re.compile(
        "(?:Re)?publicad[oa]|Revogad[oa]|Prorrogad[oa]|Alterad[oa]|Rejeitad[oa]|Consolidad[oa]|Regulamentad[oa]|"
        "Este texto não|"
        "CONSIDERANDO|"
        "Mensagem de Veto|"
        "DECRETA|"
        "RESOLVE|"
        "LIVRO|CAP[IÍ]TULO|SEÇÃO|SUBSEÇÃO|PARTE|T[ÍI]TULO|"
        "(?:Autoria|Autores|Autora?s?):"
        "",
        re.IGNORECASE,
    )

    i = first_art_ind - 2
    while i >= 1 and (
        min(len(segs[i]), len(segs[i - 1])) <= 5
        or reg_skip.match(segs[i])
        or reg_someone.match(segs[i])
        or reg_skip.match(segs[i - 1])
        or reg_someone.match(segs[i - 1])
    ):
        i -= 1

    if i >= 1:
        return segs[i - 1], segs[i]


def fn_preprocessing_ba(text: str) -> str:
    text = reg_preproc_ba_art.sub(r" Art.", text)
    text = reg_preproc_ba_law.sub(r" \1 ", text)
    text = reg_preproc_ba_whitespaces.sub(r" ", text)
    return text


def make_pairs_state_bills(
    debug: bool = False,
) -> dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    state_dirs = glob.glob(os.path.abspath("./raw_data/legislacoes_estaduais/legislacao_*"))
    state_dirs = [
        item for item in state_dirs if not re.search("_(?:pi|rr)$", os.path.basename(item))
    ]

    assert len(state_dirs) == 24

    uris_per_state = collections.defaultdict(list)
    for state_dir in state_dirs:
        state_uris = glob.glob(os.path.join(state_dir, "*"))
        state_acronym = os.path.basename(state_dir).split("_")[-1]
        uris_per_state[state_acronym] = [item for item in state_uris if "_ocr" not in item]

    assert len(uris_per_state)

    pairs = collections.defaultdict(lambda: ([], []))
    segmenter = segmentador.BERTSegmenter(device="cuda:0")
    min_length = 100

    acronym_to_name = {
        "sp": "SÃO PAULO",
        "es": "ESPÍRITO SANTO",
        "mg": "MINAS GERAIS",
        "rj": "RIO DE JANEIRO",
        "go": "GOIÁS",
        "ms": "MATO GROSSO DO SUL",
        "mt": "MATO GROSSO",
        "rs": "RIO GRANDE DO SUL",
        "sc": "SANTA CATARINA",
        "pr": "PARANÁ",
        "ba": "BAHIA",
        "pe": "PERNAMBUCO",
        "ma": "MARANHÃO",
        "rn": "RIO GRANDE DO NORTE",
        "ce": "CEARÁ",
        "se": "SERGIPE",
        "al": "ALAGOAS",
        "pb": "PARAÍBA",
        "ro": "RONDÔNIA",
        "ap": "AMAPÁ",
        "am": "AMAZÔNIA",
        "pa": "PARÁ",
        "ac": "ACRE",
        "to": "TOCANTINS",
    }

    reg_titles = re.compile(
        r"T[ií]tulo (?:de )?Cidadã|T[ií]tulo de|t[ií]tulo .{,10}de Cidadã", re.IGNORECASE
    )
    reg_public_utility = re.compile("utilidade p[uú]blica", re.IGNORECASE)

    acronym_to_filters = {
        "es": re.compile("para o fim que espec[ií]fica|(?:\|\s*){3,}", re.IGNORECASE),
        "go": re.compile(
            r"que especifica(?:.{,5}e d[aá] outras provid[eê]ncias)?\.$|(?:\|\s*){3,}",
            re.IGNORECASE,
        ),
        "pe": re.compile(
            r"PRODEPE|Abre a?o Orçamento Fiscal|Concede est[ií]mulo|incentivo fiscal", re.IGNORECASE
        ),
        "rj": re.compile(r"finalidade que menciona", re.IGNORECASE),
        "sp": re.compile(r"^(?:Denomina|Dá a denominação)", re.IGNORECASE),
    }

    acronym_to_preprocessing = {
        "ba": fn_preprocessing_ba,
        "mg": lambda x: re.sub("^\s*((?:Imprimir documento|Entenda a norma)\s*)+", "", x),
        "ro": lambda x: re.sub("LEIN", "LEI N", x),
    }

    acronym_to_fn_doc_name = collections.defaultdict(lambda: fn_doc_name_default)
    acronym_to_fn_doc_name.update(
        {
            "ac": fn_doc_name_ac,
            "al": fn_doc_name_al,
            "am": fn_doc_name_am,
            "ap": fn_doc_name_ap,
            "es": fn_doc_name_es,
            "go": fn_doc_name_go,
            "ma": fn_doc_name_ma,
            "mg": fn_doc_name_mg,
            "ms": fn_doc_name_ms,
            "mt": fn_doc_name_mt,
            "pa": fn_doc_name_pa,
            "pb": fn_doc_name_pb,
            "pe": fn_doc_name_pe,
            "pr": fn_doc_name_pr,
            "sc": fn_doc_name_sc,
            "sp": fn_doc_name_sp,
            "to": fn_doc_name_to,
        }
    )

    assert len(acronym_to_name) == len(state_dirs)

    reg_law_decree = re.compile(r"(LEI(?: COMPLEMENTAR)?|DECRETO(?: ORÇAMENTÁRIO)?)", re.IGNORECASE)

    for state_acronym, uris in uris_per_state.items():
        if debug:
            uris = uris[:10]

        pbar = tqdm.tqdm(uris, desc=f"(state leg bills, {state_acronym})", total=len(uris))
        iters_to_print = max(int(len(pbar) * 0.05), 1)
        cur_iters_to_print = iters_to_print
        pairs_leg, pairs_just = pairs[state_acronym]

        for i, uri in enumerate(pbar, 1):
            cur_iters_to_print -= 1

            with open(uri, "r", encoding="utf-8") as f_in:
                doc_content = f_in.read()

            if state_acronym in acronym_to_preprocessing:
                doc_content = acronym_to_preprocessing[state_acronym](doc_content)

            segs = segmenter(doc_content[:4000], remove_noise_subsegments=True)
            first_article, j = utils.fetch_first_item_with_index(
                segs, segmenter=segmenter, prefix="Art"
            )

            if not first_article:
                continue
            if state_acronym in acronym_to_filters and acronym_to_filters[state_acronym].search(
                first_article
            ):
                continue
            if reg_titles.search(first_article):
                continue

            first_article = utils.remove_spurious_whitespaces(first_article)
            doc_name, ementa_content = acronym_to_fn_doc_name[state_acronym](segs, j)

            if (
                not ementa_content
                or not doc_name
                or not reg_accepted_documents_name.match(doc_name)
            ):
                continue

            if state_acronym in acronym_to_filters and acronym_to_filters[state_acronym].search(
                ementa_content
            ):
                continue
            if reg_public_utility.search(first_article) or reg_public_utility.search(
                ementa_content
            ):
                continue

            doc_name = reg_law_decree.sub(r"\1 ESTADUAL", doc_name)
            doc_name = f"{acronym_to_name[state_acronym]}/{state_acronym.upper()}, {doc_name}"
            ementa_content_ext = f"({doc_name}) {ementa_content}"

            if min(len(ementa_content), len(first_article)) >= min_length:
                pairs_leg.append((ementa_content_ext, first_article))

            just_a, just_b = split_justificativa(doc_content, segmenter=segmenter)

            if just_a and min(len(ementa_content), len(just_a)) >= min_length:
                pairs_just.append((ementa_content, just_a))

            if just_b and just_a and min(len(just_b), len(just_a)) >= min_length:
                pairs_just.append((just_b, just_a))

            if cur_iters_to_print <= 0:
                cur_iters_to_print = iters_to_print
                if pairs_leg:
                    utils.print_example(*pairs_leg[-1])
                if len(pairs_just) >= 2:
                    utils.print_example(*pairs_just[-2])
                if len(pairs_just) >= 1:
                    utils.print_example(*pairs_just[-1])

    if not debug:
        assert len(pairs)
        just_cumsum = 0

        for a, b in pairs.values():
            assert len(a)
            just_cumsum += len(b)

        assert just_cumsum  # Some bills don't have 'justification'.

    return pairs
