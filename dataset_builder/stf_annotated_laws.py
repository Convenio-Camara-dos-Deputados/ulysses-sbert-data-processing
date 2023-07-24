import re
import os

import bs4
import colorama

from . import utils


def make_pairs_const(iters_to_print: int = 30) -> list[tuple[str, str]]:
    with open(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "const_comentada.html"), "r"
    ) as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    apresentacoes = parsed.find_all("div", id=re.compile(r"apresentacao[0-9]+"))
    apresentacoes.pop(0)
    last_art = ""
    pairs = []

    for i, apresentacao in enumerate(apresentacoes):
        for child in apresentacao.find_all("div", recursive=False):
            content = child.find("div", class_="conteudo", recursive=False).get_text()
            content = content.strip()

            if child["class"][0] == "art":
                last_art = content
            elif last_art:
                content = f"{last_art} (...)\n{content}"

            comments = child.find_all("div", class_="com", recursive=False)
            for comment in comments:
                title = comment.find(class_="com-titulo")
                if title:
                    title.extract()

                comment = comment.get_text()
                comment = comment.split("[")[0]
                comment = comment.replace("NOVO: ", "")
                comment = comment.replace("Nota: ", "")
                comment = comment.strip()

                pairs.append((f"(CF/1988) {content}", comment))

            if i % iters_to_print == 0 and comments:
                print(colorama.Fore.YELLOW, pairs[-1][0], colorama.Style.RESET_ALL, sep="")
                for comment in comments:
                    print(">>>", pairs[-1][1])
                print("\n\n")

    assert len(pairs)

    return pairs


def _make_pairs(urn: str, law_name: str, iters_to_print: int = 10) -> list[tuple[str, str]]:
    uri = os.path.join(utils.Config.COMPLEMENTARY_DATADIR, urn)

    with open(os.path.abspath(uri), "r") as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    items = parsed.find("section", id="conteudo").find("div").find("div")
    items = items.find_all("div", class_=re.compile(r"^(?:titulo|com)$"), recursive=False)
    last_art = ""
    pairs = []

    for i, item in enumerate(items):
        print_iter = i % iters_to_print == 0

        if item["class"][0] == "titulo":
            child = item.find("div")
            content = item.get_text().strip()
            if child["class"][0] == "ART":
                last_art = content
            else:
                content = f"{last_art} (...)\n{content}"

        else:
            comment = item.get_text()
            comment = comment.split("[")[0]
            comment = comment.strip().strip('"')
            comment = re.sub(r"\n{2,}", r"\n", comment)

            pairs.append((f"({law_name}) {content}", comment))

        if print_iter and pairs:
            utils.print_example(*pairs[-1])

    assert len(pairs)

    return pairs


def make_pairs_oab(iters_to_print: int = 10) -> list[tuple[str, str]]:
    return _make_pairs(
        urn="oab_comentada.html",
        iters_to_print=iters_to_print,
        law_name="LEI Nº 8.906/1994 - Estatuto da OAB",
    )


def make_pairs_lei_9882_1999(iters_to_print: int = 10) -> list[tuple[str, str]]:
    return _make_pairs(
        urn="lei_9882_1999_comentada.html",
        iters_to_print=iters_to_print,
        law_name="LEI Nº 9882/1999",
    )


def make_pairs_lei_9868_1999(iters_to_print: int = 10) -> list[tuple[str, str]]:
    return _make_pairs(
        urn="lei_9868_1999_comentada.html",
        iters_to_print=iters_to_print,
        law_name="LEI Nº 9869/1999",
    )
