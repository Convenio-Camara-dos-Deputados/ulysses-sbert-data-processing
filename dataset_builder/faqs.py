import typing as t
import os
import glob
import re
import json
import time
import collections

import bs4
import pandas as pd

from . import utils


def make_pairs_senado() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with open(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "senado_faq.html"), "r") as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    for section in parsed.find_all("div", class_="panel-default"):
        section_title = section.find("h4").get_text().strip()
        for question_box in section.find_all("div", class_="caixa-perguntas"):
            question = question_box.find("a", class_="link-caixa-perguntas").get_text().strip()

            answer = question_box.find("div", class_="panel-collapse").get_text().strip()
            answer = utils.reg_empty_lines.sub("", answer)

            new_pair = (f"(Senado Federal - {section_title}) {question}", answer)
            pairs.append(new_pair)

            utils.print_example(*new_pair)

    assert len(pairs)
    return pairs


def make_pairs_portal_da_transp() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with open(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "portal_da_transparencia_faq.html"), "r") as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    for section in parsed.find_all("div", class_="row"):
        try:
            section_title = section.find("h3").get_text().strip()
            print(section_title)

        except Exception:
            continue

        for q, a in zip(
            section.find_all("button", class_="collapsed"),
            section.find_all("div", class_="collapse"),
        ):
            question = q.get_text().strip()

            answer = a.get_text().strip()
            answer = answer.replace("\n", "")
            answer = re.sub(r"([:;.])\s*-", r"\1\n-", answer)
            answer = re.sub(r"([.;:])([a-záéíóúâêôãẽõüçà])", r"\1 \2", answer, flags=re.IGNORECASE)
            answer = "\n".join(utils.natural_sentence_tokenize(answer))
            answer = re.sub(r"([0-9]\.)\s*\n", r"\n\1 ", answer)

            new_pair = (
                f"(Portal da Transparência - {section_title}) {question}",
                answer,
            )

            pairs.append(new_pair)

            utils.print_example(*new_pair)

    assert len(pairs)
    return pairs


def make_pairs_receita_federal() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with open(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "receita_federal_faq.html"), "r") as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    for section in parsed.find_all("div", class_="content"):
        section_title = section.find("h1").get_text().strip()
        for answer in section.find_all("div", class_="conteudo"):
            question = answer.find_previous_sibling("a").get_text().strip()

            answer = answer.get_text().strip()
            answer = utils.reg_empty_lines.sub("", answer)
            answer = re.sub(r"Resposta elaborada em:[^\n]*\n", "", answer)

            new_pair = (f"(Receita Federal - {section_title}) {question}", answer)
            pairs.append(new_pair)

            utils.print_example(*new_pair)

    assert len(pairs)
    return pairs


def make_pairs_bc() -> list[tuple[str, str]]:
    df = pd.read_csv(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "bc_faqs.tsv"), sep="\t", index_col=0)
    df = df.applymap(lambda x: re.sub("^Saiba mais[^\n]*$", "", x, flags=re.MULTILINE))
    pairs = df.values.tolist()
    assert len(pairs)
    return pairs


def _make_pairs_gov_standard_html(input_uri: str, source: t.Optional[str] = None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    with open(os.path.abspath(input_uri), "r") as f_in:
        parsed = bs4.BeautifulSoup(f_in.read(), "html.parser")

    if source:
        source = f"{source} - "
    else:
        source = ""

    for section in parsed.find_all("div", class_="content"):
        section_title = section.find("h1").get_text().strip()
        if section_title.lower() == "perguntas frequentes":
            section_title = ""

        for subsection in section.select("div > ul > li", recursive=False):
            try:
                subsection_title = subsection.find("a", class_="toggle").get_text().strip()
            except AttributeError:
                continue

            if subsection_title == section_title:
                subsection_title = ""
            else:
                subsection_title = f" - {subsection_title}"

            for item in subsection.select("li > ul > li", recursive=False):
                question = item.find("a").get_text().strip()

                question = re.sub(r"^\d+[-.]\s*", "", question)

                answer = item.find("div", class_="conteudo").get_text().strip()
                answer = utils.reg_empty_lines.sub("", answer)

                new_pair = (
                    f"({source if section_title else source.split(' - ')[0]}{section_title}{subsection_title}) {question}",
                    answer,
                )
                pairs.append(new_pair)

                utils.print_example(*new_pair)

    assert len(pairs)
    return pairs


def make_pairs_mec() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "min_ed_faqs.html"), source="MEC")


def make_pairs_midr() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "min_int_e_dev_reg_faqs.html"),
        source="MDR",
    )


def make_pairs_mdh() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "min_dir_hum.html"), source="MDH")


def make_pairs_mds() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "mds_faq.html"), source="MDH")


def make_pairs_defesa() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "defesa_faq.html"), source="MDH")


def make_pairs_cidades() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "cidades_faq.html"), source="CIDADES")


def make_pairs_mcom() -> list[tuple[str, str]]:
    return _make_pairs_gov_standard_html(os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "mcom_faq.html"), source="MCOM")


def make_pairs_general() -> dict[str, list[tuple[str, str]]]:
    with open(os.path.join(utils.Config.TESEMO_PATH, "outros/o11_faqs.txt"), "r") as f_in:
        text = f_in.read()

    pairs: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    reg_tag = re.compile(r"^\(([^\)]+)\)")

    for i, pair in enumerate(re.split(r"\n{2}(?=\()", text)):
        question, *answer = pair.split("\n")

        question = question.strip()
        answer = "\n".join(answer).strip()

        tag = reg_tag.search(question).group(1)
        new_pair = (question, answer)

        pairs[tag].append(new_pair)

        if i % 500 == 0:
            utils.print_example(*new_pair)

    assert len(pairs)

    return pairs
