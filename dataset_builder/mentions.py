import os
import glob

import regex as re

from . import utils


def make_pairs():
    reg_noise = re.compile(r"Texto compilado|Mensagem de veto|Vigência|Regulamento")

    paths = glob.glob(
        os.path.join(utils.Config.TESEMO_PATH, "legislativo", "l9_codigos_legais/*.txt")
    )
    paths.extend(
        glob.glob(os.path.join(utils.Config.TESEMO_PATH, "legislativo", "l12_estatutos/*.txt"))
    )
    paths.append(
        os.path.join(
            utils.Config.TESEMO_PATH,
            "legislativo",
            "l13_constituicao_da_republica_federativa_do_brasil_de_1988.txt",
        )
    )

    reg_item_law = re.compile(r"^(.*)\s+(\([^\)]{10,}\))$", re.MULTILINE)
    reg_peloa = re.compile(r".*(?:pel[oa]|vide)\s*", re.IGNORECASE)
    reg_revogado = re.compile("revogad[ao]|vetad[ao]", re.IGNORECASE)
    reg_encerrado = re.compile(
        r"Produção de efeito|Vigência encerrada|Regulamento|reciprocidade|\bouro\b|Congresso Nacional",
        re.IGNORECASE,
    )

    pairs = []

    for uri in paths:
        with open(uri, "r") as f_in:
            items = [item.strip() for item in f_in.readlines()]

        for item in items:
            aux = []
            a = item
            match_ = reg_item_law.match(a)
            while match_:
                a = match_.group(1).strip()
                b = match_.group(2).strip().lstrip("(").rstrip(")")
                aux.insert(0, reg_peloa.sub("", b))
                match_ = reg_item_law.match(a)

            if aux and not reg_revogado.search(a) and len(a) > 10:
                for subitem in aux:
                    if (
                        len(subitem) > 10
                        and not reg_encerrado.search(subitem)
                        and ("A" <= subitem[0] <= "Z")
                    ):
                        pairs.append((a, subitem))

    assert len(pairs)

    return pairs
