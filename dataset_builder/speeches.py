import glob
import os
import re

import tqdm

from . import utils


def make_pairs(long_segments: bool = False) -> list[tuple[str, str]]:
    base_dir = os.path.join(utils.Config.TESEMO_PATH, "legislativo", "l4_discursos_da_camara_dos_deputados")
    summary_uris = glob.glob(os.path.join(base_dir, "sumarios_discursos_subset_a", "*.txt"))

    assert len(summary_uris)

    reg_split = re.compile(
        r"^\s*[AO]?\s+SRA?\.[^\(]{1,70}(?:\([^\)]{1,70}\))?\s*-\s*|(?:^|\n)[\sA-ZÇ0-9ÁÀÂÃÉẼÊÔÕÓÚŨÜÍ]{15,}\n",
        re.MULTILINE,
    )
    reg_spaces = re.compile(r"^\s+", re.MULTILINE)
    reg_start_noise = re.compile(r"^[^a-zçáàãâéẽêíóõôúü]+", re.IGNORECASE)

    pairs: list[tuple[str, str]] = []

    pbar = tqdm.tqdm(summary_uris)
    hits = 0

    iters_to_print: int | float
    if utils.Config.IT_TO_PRINT:
        iters_to_print = int(utils.Config.IT_TO_PRINT * len(pbar))
    else:
        iters_to_print = float("+inf")

    hits_to_print = iters_to_print
    dir_speech_content = os.path.join(base_dir, "conteudo_discursos_subset_a")

    for i, summ_uri in enumerate(pbar, 1):
        with open(summ_uri, "r", encoding="utf-8") as f_in:
            summ_content = f_in.read()

        if long_segments:
            sents_summ = [summ_content]
        else:
            sents_summ = utils.natural_sentence_tokenize(summ_content)

        if not sents_summ or "Discurso proferido" in sents_summ[0]:
            continue

        content_uri = os.path.join(dir_speech_content, os.path.basename(summ_uri))
        with open(content_uri, "r", encoding="utf-8") as f_in:
            content = reg_start_noise.sub("", reg_spaces.sub("", f_in.read()))

        if long_segments:
            sents_cont = [content[:10000]]

        else:
            sents_cont = reg_split.split(content)

            if not sents_cont[0]:
                sents_cont.pop(0)

            aux = sents_cont[0].split("\n")
            k = 3 if sum(map(len, aux[:3])) < 1200 else 2
            sents_cont[0] = "\n".join(aux[:k])

        if min(len(sents_cont[0]), len(sents_summ[0])) <= 80:
            continue

        hits += 1
        pbar.set_description(f"(speeches to pairs) hit rate: {100 * hits / i:.2f}%")

        item = (sents_cont[0].replace("\t", " "), sents_summ[0])
        pairs.append(item)

        hits_to_print -= 1
        if hits_to_print <= 0:
            utils.print_example(*item)
            hits_to_print = iters_to_print

    assert len(pairs)

    return pairs
