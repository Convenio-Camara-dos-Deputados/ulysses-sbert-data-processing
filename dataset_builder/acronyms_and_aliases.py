import os
import itertools

import pandas as pd

from . import utils


def _fetch_doc_contents(uri: str, segmenter) -> str:
    with open(uri, "r", encoding="utf-8") as f_in:
        text = f_in.read(8000)

    segs = segmenter(text)
    i = utils.fetch_first_item_index(segs)
    return "\n".join(segs[i:])


def make_pairs(long_segments: bool = False):
    pairs = []

    if long_segments:
        import segmentador

        segmenter = segmentador.BERTSegmenter(device="cuda:0")

        doc_urn_to_uri = {
            "codigo_aeronautica": "l9_codigos_legais/C_digo_Brasileiro_de_Aeron_utica.txt",
            "codigo_telecomunicacao": "l9_codigos_legais/C_digo_Brasileiro_de_Telecomunica_es.txt",
            "codigo_civil": "l9_codigos_legais/C_digo_Civil.txt",
            "codigo_comercial": "l9_codigos_legais/C_digo_Comercial.txt",
            "codigo_conduta_alta_adm": "l9_codigos_legais/C_digo_de_Conduta_da_Alta_Administra_o_Federal.txt",
            "codigo_consumidor": "l9_codigos_legais/C_digo_de_Defesa_do_Consumidor.txt",
            "codigo_aguas": "l9_codigos_legais/C_digo_de_guas.txt",
            "codigo_minas": "l9_codigos_legais/C_digo_de_Minas.txt",
            "codigo_processo_civil": "l9_codigos_legais/C_digo_de_Processo_Civil.txt",
            "codigo_processo_penal": "l9_codigos_legais/C_digo_de_Processo_Penal.txt",
            "codigo_processo_penal_militar": "l9_codigos_legais/C_digo_de_Processo_Penal_Militar.txt",
            "novo_codigo_processo_civil": "l9_codigos_legais/Novo_C_digo_de_Processo_Civil.txt",
            "codigo_transito": "l9_codigos_legais/C_digo_de_Tr_nsito_Brasileiro.txt",
            "codigo_eleitoral": "l9_codigos_legais/C_digo_Eleitoral.txt",
            "codigo_florestal": "l9_codigos_legais/C_digo_Florestal.txt",
            "codigo_penal": "l9_codigos_legais/C_digo_Penal.txt",
            "codigo_penal_militar": "l9_codigos_legais/C_digo_Penal_Militar.txt",
            "codigo_tributario": "l9_codigos_legais/C_digo_Tribut_rio_Nacional.txt",
            "codigo_trabalho": "l9_codigos_legais/Consolida_o_das_Leis_do_Trabalho.txt",
            "estatuto_oab": "l12_estatutos/Estatuto_da_Advocacia_e_da_Ordem_dos_Advogados_do_Brasil.txt",
            "estatuto_cidade": "l12_estatutos/Estatuto_da_Cidade.txt",
            "estatuto_crianca": "l12_estatutos/Estatuto_da_Crian_a_e_do_Adolescente.txt",
            "estatuto_racial": "l12_estatutos/Estatuto_da_Igualdade_Racial.txt",
            "estatuto_juventude": "l12_estatutos/Estatuto_da_Juventude.txt",
            "estatuto_metropole": "l12_estatutos/Estatuto_da_Metr_pole.txt",
            "estatuto_cancer": "l12_estatutos/Estatuto_da_Pessoa_com_C_ncer.txt",
            "estatuto_deficiencia": "l12_estatutos/Estatuto_da_Pessoa_com_Defici_ncia.txt",
            "estatuto_terra": "l12_estatutos/Estatuto_da_Terra.txt",
            "estatuto_torcedor": "l12_estatutos/Estatuto_de_Defesa_do_Torcedor.txt",
            "estatuto_desarmamento": "l12_estatutos/Estatuto_do_Desarmamento.txt",
            "estatuto_estrangeiro": "l12_estatutos/Estatuto_do_Estrangeiro.txt",
            "estatuto_idoso": "l12_estatutos/Estatuto_do_Idoso.txt",
            "estatuto_indio": "l12_estatutos/Estatuto_do_ndio.txt",
            "estatuto_militares": "l12_estatutos/Estatuto_dos_Militares.txt",
            "estatuto_museus": "l12_estatutos/Estatuto_dos_Museus.txt",
            "estatuto_refugiados": "l12_estatutos/Estatuto_dos_Refugiados.txt",
            "estatuto_empresa": "l12_estatutos/Estatuto_Nacional_da_Microempresae_da_Empresa_de_Pequeno_Porte.txt",
        }

        doc_urn_to_uri = {
            k: os.path.join(utils.Config.TESEMO_PATH, "legislativo", v)
            for k, v in doc_urn_to_uri.items()
        }
        fn_fetch = lambda doc_urn: _fetch_doc_contents(doc_urn_to_uri[doc_urn], segmenter=segmenter)

    segs = [
        "Estatuto da Advocacia e da Ordem dos Advogados do Brasil",
        "LEI Nº 8. 906, DE 4 DE JULHO DE 1994. ( Vide ADIN 6278 )",
        "Dispõe sobre o Estatuto da Advocacia e a Ordem dos Advogados do Brasil ( OAB ).",
        "Art. 1º São atividades privativas de advocacia :"
        if not long_segments
        else fn_fetch("estatuto_oab"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Criança e do Adolescente",
        "LEI Nº 8. 069, DE 13 DE JULHO DE 1990. ( Vide Lei nº 14. 344, de 2022 )",
        "Dispõe sobre o Estatuto da Criança e do Adolescente e dá outras providências.",
        "Art. 1º Esta Lei dispõe sobre a proteção integral à criança e ao adolescente."
        if not long_segments
        else fn_fetch("estatuto_crianca"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Cidade",
        "LEI No 10. 257, DE 10 DE JULHO DE 2001.",
        "Regulamenta os arts. 182 e 183 da Constituição Federal, estabelece diretrizes gerais da política urbana e dá outras providências.",
        "Art. 1o Na execução da política urbana, de que tratam os arts. 182 e 183 da Constituição Federal, será aplicado o previsto nesta Lei."
        if not long_segments
        else fn_fetch("estatuto_cidade"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto de Defesa do Torcedor",
        "LEI No 10. 671, DE 15 DE MAIO DE 2003.",
        "Dispõe sobre o Estatuto de Defesa do Torcedor e dá outras providências.",
        "Art. 1o Este Estatuto estabelece normas de proteção e defesa do torcedor."
        if not long_segments
        else fn_fetch("estatuto_torcedor"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto do Desarmamento",
        "LEI No 10. 826, DE 22 DE DEZEMBRO DE 2003.",
        "Dispõe sobre registro, posse e comercialização de armas de fogo e munição, sobre o Sistema Nacional de Armas – Sinarm, define crimes e dá outras providências.",
        "Art. 1o O Sistema Nacional de Armas – Sinarm, instituído no Ministério da Justiça, no âmbito da Polícia Federal, tem circunscrição em todo o território nacional."
        if not long_segments
        else fn_fetch("estatuto_desarmamento"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto do Estrangeiro",
        "LEI Nº 13. 445, DE 24 DE MAIO DE 2017.",
        "Institui a Lei de Migração.",
        "Art. 1º Esta Lei dispõe sobre os direitos e os deveres do migrante e do visitante, regula a sua entrada e estada no País e estabelece princípios e diretrizes para as políticas públicas para o emigrante."
        if not long_segments
        else fn_fetch("estatuto_estrangeiro"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto do Idoso",
        "LEI No 10. 741, DE 1º DE OUTUBRO DE 2003. ( Vide Decreto nº 6. 214, de 2007 )",
        "Dispõe sobre o Estatuto do Idoso e dá outras providências.",
        "Art. 1o É instituído o Estatuto do Idoso, destinado a regular os direitos assegurados às pessoas com idade igual ou superior a 60 ( sessenta ) anos."
        if not long_segments
        else fn_fetch("estatuto_idoso"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Igualdade Racial",
        "LEI Nº 12. 288, DE 20 DE JULHO DE 2010. ( Vide Decreto nº 8. 136, de 2013 )",
        "Institui o Estatuto da Igualdade Racial ; altera as Leis nos 7. 716, de 5 de janeiro de 1989, 9. 029, de 13 de abril de 1995, 7. 347, de 24 de julho de 1985, e 10. 778, de 24 de novembro de 2003.",
        "Art. 1o Esta Lei institui o Estatuto da Igualdade Racial, destinado a garantir à população negra a efetivação da igualdade de oportunidades, a defesa dos direitos étnicos individuais, coletivos e difusos e o combate à discriminação e às demais formas de intolerância étnica."
        if not long_segments
        else fn_fetch("estatuto_racial"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto do índio",
        "LEI Nº 6. 001, DE 19 DE DEZEMBRO DE 1973.",
        "Dispõe sobre o Estatuto do Índio.",
        "Art. 1º Esta Lei regula a situação jurídica dos índios ou silvícolas e das comunidades indígenas, com o propósito de preservar a sua cultura e integrá - los, progressiva e harmoniosamente, à comunhão nacional."
        if not long_segments
        else fn_fetch("estatuto_indio"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Juventude",
        "LEI Nº 12. 852, DE 5 DE AGOSTO DE 2013. ( Vide Decreto nº 9. 306, de 2018 )",
        "Institui o Estatuto da Juventude e dispõe sobre os direitos dos jovens, os princípios e diretrizes das políticas públicas de juventude e o Sistema Nacional de Juventude - SINAJUVE.",
        "Art. 1º Esta Lei institui o Estatuto da Juventude e dispõe sobre os direitos dos jovens, os princípios e diretrizes das políticas públicas de juventude e o Sistema Nacional de Juventude - SINAJUVE."
        if not long_segments
        else fn_fetch("estatuto_juventude"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Metrópole",
        "LEI Nº 13. 089, DE 12 DE JANEIRO DE 2015.",
        "Institui o Estatuto da Metrópole, altera a Lei nº 10. 257, de 10 de julho de 2001, e dá outras providências.",
        "Art. 1º Esta Lei, denominada Estatuto da Metrópole, estabelece diretrizes gerais para o planejamento, a gestão e a execução das funções públicas de interesse comum em regiões metropolitanas e em aglomerações urbanas instituídas pelos Estados, normas gerais sobre o plano de desenvolvimento urbano integrado e outros instrumentos de governança interfederativa, e critérios para o apoio da União a ações que envolvam governança interfederativa no campo do desenvolvimento urbano, com base nos i ncisos XX do art. 21, IX do art. 23 e I do art. 24, no § 3º do art. 25 e no art. 182 da Constituição Federal."
        if not long_segments
        else fn_fetch("estatuto_metropole"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto dos Militares",
        "LEI Nº 6. 880, DE 9 DE DEZEMBRO DE 1980 ( Vide Decreto nº 88. 455, de 1983 ) ( Vide Decreto nº 4. 307, de 2002 ) ( Vide Decreto nº 4. 346, de 2002 ) ( Vide Decreto nº 10. 750, de 2021 )",
        "Dispõe sobre o Estatuto dos Militares.",
        "Art. 1º O presente Estatuto regula a situação, obrigações, deveres, direitos e prerrogativas dos membros das Forças Armadas."
        if not long_segments
        else fn_fetch("estatuto_militares"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto dos Museus",
        "LEI Nº 11. 904, DE 14 DE JANEIRO DE 2009.",
        "Institui o Estatuto de Museus e dá outras providências.",
        "Art. 1o Consideram - se museus, para os efeitos desta Lei, as instituições sem fins lucrativos que conservam, investigam, comunicam, interpretam e expõem, para fins de preservação, estudo, pesquisa, educação, contemplação e turismo, conjuntos e coleções de valor histórico, artístico, científico, técnico ou de qualquer outra natureza cultural, abertas ao público, a serviço da sociedade e de seu desenvolvimento."
        if not long_segments
        else fn_fetch("estatuto_museus"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto Nacional da Microempresa e da Empresa de Pequeno Porte",
        "LEI COMPLEMENTAR Nº 123, DE 14 DE DEZEMBRO DE 2006",
        "Institui o Estatuto Nacional da Microempresa e da Empresa de Pequeno Porte ; altera dispositivos das Leis no 8. 212 e 8. 213, ambas de 24 de julho de 1991, da Consolidação das Leis do Trabalho - CLT, aprovada pelo Decreto - Lei no 5. 452, de 1o de maio de 1943, da Lei no 10. 189, de 14 de fevereiro de 2001, da Lei Complementar no 63, de 11 de janeiro de 1990 ; e revoga as Leis no 9. 317, de 5 de dezembro de 1996, e 9. 841, de 5 de outubro de 1999.",
        "Art. 1o Esta Lei Complementar estabelece normas gerais relativas ao tratamento diferenciado e favorecido a ser dispensado às microempresas e empresas de pequeno porte no âmbito dos Poderes da União, dos Estados, do Distrito Federal e dos Municípios, especialmente no que se refere :"
        if not long_segments
        else fn_fetch("estatuto_empresa"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Pessoa com Câncer",
        "LEI Nº 14. 238, DE 19 DE NOVEMBRO DE 2021",
        "Institui o Estatuto da Pessoa com Câncer ; e dá outras providências.",
        "Art. 1º Fica instituído o Estatuto da Pessoa com Câncer, destinado a assegurar e a promover, em condições de igualdade, o acesso ao tratamento adequado e o exercício dos direitos e das liberdades fundamentais da pessoa com câncer, com vistas a garantir o respeito à dignidade, à cidadania e à sua inclusão social."
        if not long_segments
        else fn_fetch("estatuto_cancer"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Pessoa com Deficiência",
        "LEI Nº 13. 146, DE 6 DE JULHO DE 2015.",
        "Institui a Lei Brasileira de Inclusão da Pessoa com Deficiência ( Estatuto da Pessoa com Deficiência ).",
        "Art. 1º É instituída a Lei Brasileira de Inclusão da Pessoa com Deficiência ( Estatuto da Pessoa com Deficiência ), destinada a assegurar e a promover, em condições de igualdade, o exercício dos direitos e das liberdades fundamentais por pessoa com deficiência, visando à sua inclusão social e cidadania."
        if not long_segments
        else fn_fetch("estatuto_deficiencia"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto dos Refugiados",
        "LEI Nº 9. 474, DE 22 DE JULHO DE 1997.",
        "Define mecanismos para a implementação do Estatuto dos Refugiados de 1951, e determina outras providências.",
        "Art. 1º Será reconhecido como refugiado todo indivíduo que :"
        if not long_segments
        else fn_fetch("estatuto_refugiados"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Estatuto da Terra",
        "LEI Nº 4. 504, DE 30 DE NOVEMBRO DE 1964.",
        "Dispõe sobre o Estatuto da Terra, e dá outras providências.",
        "Art. 1° Esta Lei regula os direitos e obrigações concernentes aos bens imóveis rurais, para os fins de execução da Reforma Agrária e promoção da Política Agrícola."
        if not long_segments
        else fn_fetch("estatuto_terra"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Civil",
        "LEI N o 10. 406, DE 10 DE JANEIRO DE 2002",
        "Lei de Introdução às normas do Direito Brasileiro ( Vide Lei nº 14. 195, de 2021 )",
        "Institui o Código Civil.",
        "Art. 1º Toda pessoa é capaz de direitos e deveres na ordem civil."
        if not long_segments
        else fn_fetch("codigo_civil"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Processo Civil",
        "LEI N o 5. 869, DE 11 DE JANEIRO DE 1973. Revogada pela Lei nº 13. 105, de 2015",
        "Institui o Código de Processo Civil.",
        "Art. 1 o A jurisdição civil, contenciosa e voluntária, é exercida pelos juízes, em todo o território nacional, conforme as disposições que este Código estabelece."
        if not long_segments
        else fn_fetch("codigo_processo_civil"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Novo Código de Processo Civil",
        "LEI Nº 13. 105, DE 16 DE MARÇO DE 2015.",
        "Art. 1º O processo civil será ordenado, disciplinado e interpretado conforme os valores e as normas fundamentais estabelecidos na Constituição da República Federativa do Brasil, observando - se as disposições deste Código."
        if not long_segments
        else fn_fetch("novo_codigo_processo_civil"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Penal",
        "DECRETO - LEI No 2. 848, DE 7 DE DEZEMBRO DE 1940 ( Vide Lei nº 1. 521, de 1951 ) ( Vide Lei nº 5. 741, de 1971 ) ( Vide Lei nº 5. 988, de 1973 ) ( Vide Lei nº 6. 015, de 1973 ) ( Vide Lei nº 6. 404, de 1976 ) ( Vide Lei nº 6. 515, de 1977 ) ( Vide Lei nº 6. 538, de 1978 ) ( Vide Lei nº 6. 710, de 1979 ) ( Vide Lei nº 7. 492, de 1986 ) ( Vide Lei nº 8. 176, de 1991 ) ( Vide Lei nº 14. 344, de 2022 )",
        "Art. 1° Não há crime sem lei anterior que o defina. Não há pena sem prévia cominação legal."
        if not long_segments
        else fn_fetch("codigo_penal"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Processo Penal",
        "DECRETO - LEI Nº 3. 689, DE 3 DE OUTUBRO DE 1941.",
        "Art. 1o O processo penal reger - se - á, em todo o território brasileiro, por este Código, ressalvados :"
        if not long_segments
        else fn_fetch("codigo_processo_penal"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Tributário Nacional",
        "LEI Nº 5. 172, DE 25 DE OUTUBRO DE 1966.",
        "Denominado Código Tributário Nacional ( Vide Decreto - lei nº 82, de 1966 ) ( Vide Decreto nº 6. 306, de 2007 )",
        "Dispõe sobre o Sistema Tributário Nacional e institui normas gerais de direito tributário aplicáveis à União, Estados e Municípios.",
        "Art. 1º Esta Lei regula, com fundamento na Emenda Constitucional nº 18, de 1º de dezembro de 1965, o sistema tributário nacional e estabelece, com fundamento no artigo 5º, inciso XV, alínea b, da Constituição Federal, as normas gerais de direito tributário aplicáveis à União, aos Estados, ao Distrito Federal e aos Municípios, sem prejuízo da respectiva legislação complementar, supletiva ou regulamentar."
        if not long_segments
        else fn_fetch("codigo_tributario"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Consolida o das Leis do Trabalho",
        "DECRETO - LEI Nº 5. 452, DE 1º DE MAIO DE 1943 ( Vide Decreto - Lei nº 127, de 1967 ) ( Vide Medida Provisória nº 1. 109, de 2022 )",
        "Aprova a Consolidação das Leis do Trabalho.",
        "Art. 1º Fica aprovada a Consolidação das Leis do Trabalho, que a este decreto - lei acompanha, com as alterações por ela introduzidas na legislação vigente."
        if not long_segments
        else fn_fetch("codigo_trabalho"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Defesa do Consumidor",
        "LEI Nº 8. 078, DE 11 DE SETEMBRO DE 1990. ( Vide Decreto nº 2. 181, de 1997 ) ( Vide pela Lei nº 13. 425, de 2017 ) ( Vide Decreto nº 11. 034, de 2022 )",
        "Dispõe sobre a proteção do consumidor e dá outras providências.",
        "Art. 1° O presente código estabelece normas de proteção e defesa do consumidor, de ordem pública e interesse social, nos termos dos arts. 5°, inciso XXXII, 170, inciso V, da Constituição Federal e art. 48 de suas Disposições Transitórias."
        if not long_segments
        else fn_fetch("codigo_consumidor"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Trânsito Brasileiro",
        "LEI Nº 9. 503, DE 23 DE SETEMBRO DE 1997 ( Vide Lei nº 14. 304, de 2022 )",
        "Institui o Código de Trânsito Brasileiro.",
        "Art. 1º O trânsito de qualquer natureza nas vias terrestres do território nacional, abertas à circulação, rege - se por este Código."
        if not long_segments
        else fn_fetch("codigo_transito"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Eleitoral",
        "LEI Nº 4. 737, DE 15 DE JULHO DE 1965",
        "Institui o Código Eleitoral.",
        "Art. 1º Este Código contém normas destinadas a assegurar a organização e o exercício de direitos políticos precipuamente os de votar e ser votado."
        if not long_segments
        else fn_fetch("codigo_eleitoral"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Florestal",
        "LEI Nº 12. 651, DE 25 DE MAIO DE 2012. ( Vide ADIN 4937 ) ( Vide ADIN 4901 )",
        "Dispõe sobre a proteção da vegetação nativa ; altera as Leis nºs 6. 938, de 31 de agosto de 1981, 9. 393, de 19 de dezembro de 1996, e 11. 428, de 22 de dezembro de 2006 ; revoga as Leis nºs 4. 771, de 15 de setembro de 1965, e 7. 754, de 14 de abril de 1989, e a Medida Provisória nº 2. 166 - 67, de 24 de agosto de 2001 ; e dá outras providências."
        if not long_segments
        else fn_fetch("codigo_florestal"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Águas",
        "DECRETO Nº 24. 643, DE 10 DE JULHO DE 1934. ( Vide Decreto - Lei nº 852, de 1938 ) ( Vide Decreto - lei nº 3. 763, de 1941 ) ( Vide Decreto nº 2. 869, de 1998 )",
        "Decreta o Código de Águas.",
        "Art. 1º As águas públicas podem ser de uso comum ou dominicais."
        if not long_segments
        else fn_fetch("codigo_aguas"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Minas",
        "DECRETO - LEI Nº 227, DE 28 DE FEVEREIRO DE 1967. ( Vide Decreto nº 62. 934, de 1968 )",
        "Dá nova redação ao Decreto - lei nº 1. 985, de 29 de janeiro de 1940. ( Código de Minas )",
        "Art. 1º Compete à União administrar os recursos minerais, a indústria de produção mineral e a distribuição, o comércio e o consumo de produtos minerais."
        if not long_segments
        else fn_fetch("codigo_minas"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Penal Militar",
        "DECRETO - LEI Nº 1. 001, DE 21 DE OUTUBRO DE 1969.",
        "Código Penal Militar Os Ministros da Marinha de Guerra, do Exército e da Aeronáutica Militar, usando das atribuições que lhes confere o art. 3º do Ato Institucional nº 16, de 14 de outubro de 1969, combinado com o § 1° do art. 2°, do Ato Institucional n° 5, de 13 de dezembro de 1968, decretam :",
        "Art. 1º Não há crime sem lei anterior que o defina, nem pena sem prévia cominação legal."
        if not long_segments
        else fn_fetch("codigo_penal_militar"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código de Processo Penal Militar",
        "DECRETO - LEI Nº 1. 002, DE 21 DE OUTUBRO DE 1969. Código de Processo Penal Militar Os Ministros da Marinha de Guerra, do Exército e da Aeronáutica Militar, usando das atribuições que lhes confere o art. 3º do Ato Institucional nº 16, de 14 de outubro de 1969, combinado com o § 1º do art. 2º do Ato Institucional n° 5, de 13 de dezembro de 1968, decretam :",
        "CÓDIGO DE PROCESSO PENAL MILITAR",
        "Art. 1º O processo penal militar reger - se - á pelas normas contidas neste Código, assim em tempo de paz como em tempo de guerra, salvo legislação especial que lhe fôr estritamente aplicável. Divergência de normas"
        if not long_segments
        else fn_fetch("codigo_processo_penal_militar"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Brasileiro de Aeron utica",
        "LEI Nº 7. 565, DE 19 DE DEZEMBRO DE 1986. ( Vide Decreto nº 95. 218, de 1987 ) ( Vide Decreto nº 3. 439, de 2000 ) ( Vide Lei nº 12. 432, de 2011 ) ( Vide Decreto nº 8. 265, de 2014 )",
        "Dispõe sobre o Código Brasileiro de Aeronáutica.",
        "Art. 1° O Direito Aeronáutico é regulado pelos Tratados, Convenções e Atos Internacionais de que o Brasil seja parte, por este Código e pela legislação complementar."
        if not long_segments
        else fn_fetch("codigo_aeronautica"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Brasileiro de Telecomunicações",
        "LEI Nº 4. 117, DE 27 DE AGOSTO DE 1962. ( Vide Lei nº 9. 472, de 1997 ) ( Vide Decreto nº 3. 965, de 2001 ) ( Vide Decreto de 16. 12 2014 ) Vide Decreto nº 2. 197, de 1997 Vide Lei nº 9. 612, de 1997",
        "Institui o Código Brasileiro de Telecomunicações.",
        "Art. 1º Os serviços de telecomunicações em todo o território do País, inclusive águas territoriais e espaço aéreo, assim como nos lugares em que princípios e convenções internacionais lhes reconheçam extraterritorialidade obedecerão aos preceitos da presente lei e aos regulamentos baixados para a sua execução."
        if not long_segments
        else fn_fetch("codigo_telecomunicacao"),
    ]
    pairs.extend(itertools.combinations(segs, 2))
    segs = [
        "Código Comercial",
        "LEI Nº 556, DE 25 DE JUNHO DE 1850.",
        "Ordem do Juízo no processo comercial ( Vide Lei nº 1. 237, de 1864 ) ( Vide Decreto - lei n° 1. 608, de 1939 ) ( Vide Lei n° 5. 869, de 11. 1. 1973 )",
        "Art. 1 - Podem comerciar no Brasil :"
        if not long_segments
        else fn_fetch("codigo_comercial"),
    ]
    pairs.extend(itertools.combinations(segs, 2))

    pairs.append(
        (
            "CONSTITUIÇÃO DA REPÚBLICA FEDERATIVA DO BRASIL DE 1988",
            "Nós, representantes do povo brasileiro, reunidos em Assembléia Nacional Constituinte para instituir um Estado Democrático, destinado a assegurar o exercício dos direitos sociais e individuais, a liberdade, a segurança, o bem-estar, o desenvolvimento, a igualdade e a justiça como valores supremos de uma sociedade fraterna, pluralista e sem preconceitos, fundada na harmonia social e comprometida, na ordem interna e internacional, com a solução pacífica das controvérsias, promulgamos, sob a proteção de Deus, a seguinte CONSTITUIÇÃO DA REPÚBLICA FEDERATIVA DO BRASIL.",
        )
    )

    if not long_segments:
        pairs.append(
            (
                "CONSTITUIÇÃO DA REPÚBLICA FEDERATIVA DO BRASIL DE 1988",
                "Art. 1º A República Federativa do Brasil, formada pela união indissolúvel dos Estados e Municípios e do Distrito Federal, constitui-se em Estado Democrático de Direito e tem como fundamentos:",
            )
        )
    else:
        with open(
            os.path.join(
                utils.Config.TESEMO_PATH,
                "legislativo",
                "l13_constituicao_da_republica_federativa_do_brasil_de_1988.txt",
            ),
            "r",
        ) as f_in:
            segs = f_in.readlines(10000)

        pairs.append(("CONSTITUIÇÃO DA REPÚBLICA FEDERATIVA DO BRASIL DE 1988", "".join(segs[21:])))

    df = pd.read_csv(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "apelidos_de_lei.csv"),
        usecols=["Nome", "Apelido"],
    )
    pairs.extend(df.values.tolist())

    df = pd.read_csv(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "apelidos_de_lei.csv"),
        usecols=["Nome", "Indexacao"],
    )
    pairs.extend(df.values.tolist())

    df = pd.read_csv(
        os.path.join(utils.Config.COMPLEMENTARY_DATADIR, "senado_leis_and_ementas.tsv"),
        index_col=0,
        sep="\t",
    )
    pairs.extend(df.values.tolist())

    pairs.extend(
        [
            ("CCJC", "Comissão de Constituição e Justiça e de Cidadania"),
            ("CCJ", "Comissão de Constituição e Justiça e de Cidadania"),
            ("CF", "Constituição Federal"),
            ("CF/88", "Constituição Federal"),
            ("CF/1988", "Constituição Federal"),
            ("(Termo legislativo) DVS", "Destaque para votação em separado"),
            ("(Termo legislativo) NR", "Nova redação"),
            ("(Termo legislativo) ONG", "Organização não-governamental"),
            ("(Termo legislativo) MP", "Medida provisória"),
            ("(Termo legislativo) CPI", "Comissão parlamentar de inquérito"),
            ("(Termo legislativo) PR", "Presidência da República"),
            ("(Termo legislativo) CN", "Congresso Nacional"),
            ("(Termo legislativo) CD", "Câmara dos Deputados"),
            ("(Termo legislativo) RICD", "Regimento Interno da Câmara dos Deputados"),
            ("(Termo legislativo) TCU", "Tribunal de Contas da União"),
            ("(Termo legislativo) LDO", "Lei de Diretrizes Orçamentárias"),
            (
                "(Termo legislativo) COM",
                "Consulta solicitada por parlamentar a comissão técnica, relativa a aspecto de constitucionalidade, juridicidade ou adequação financeira ou orçamentária",
            ),
            ("(Termo legislativo) DCR", "Denúncia por crime de responsabilidade"),
            ("(Termo legislativo) DEN", "Denúncia"),
            ("(Termo legislativo) DTQ", "Destaque"),
            ("(Termo legislativo) DVS", "Destaque para Votação em Separado"),
            ("(Termo legislativo) DVT", "Declaração de Voto"),
            ("(Termo legislativo) EMC", "Emenda Apresentada na Comissão"),
            ("(Termo legislativo) EMD", "Emenda"),
            ("(Termo legislativo) EML", "Emenda à LDO"),
            ("(Termo legislativo) EMO", "Emenda ao Orçamento"),
            ("(Termo legislativo) EMP", "Emenda de Plenário"),
            ("(Termo legislativo) EMR", "Emenda de Relator"),
            ("(Termo legislativo) Sem", "Emenda/Substitutivo do Senado"),
            ("(Termo legislativo) ERD", "Emenda de Redação"),
            ("(Termo legislativo) ESB", "Emenda ao Substitutivo"),
            ("(Termo legislativo) EXP", "Exposição"),
            ("(Termo legislativo) INA", "Indicação de Autoridade"),
            ("(Termo legislativo) INC", "Indicação"),
            ("(Termo legislativo) MPV", "Medida Provisória"),
            ("(Termo legislativo) MSC", "Mensagem"),
            ("(Termo legislativo) PAR", "Parecer de Comissão"),
            ("(Termo legislativo) PDC", "Projeto de Decreto Legislativo"),
            ("(Termo legislativo) PEC", "Proposta de Emenda à Constituição"),
            ("(Termo legislativo) PET", "Petição"),
            ("(Termo legislativo) PFC", "Proposta de Fiscalização e Controle"),
            ("(Termo legislativo) PL", "Projeto de Lei"),
            ("(Termo legislativo) PLP", "Projeto de Lei Complementar"),
            ("(Termo legislativo) PLV", "Projeto de Lei de Conversão"),
            ("(Termo legislativo) PRC", "Projeto de Resolução (CD)"),
            ("(Termo legislativo) PRF", "Projeto de Resolução do Senado Federal"),
            ("(Termo legislativo) PRN", "Projeto de Resolução (CN)"),
            ("(Termo legislativo) PRO", "Proposta"),
            (
                "(Termo legislativo) RCP",
                "Requerimento de Instituição de Comissão Parlamentar de Inquérito",
            ),
            ("(Termo legislativo) REC", "Recurso"),
            ("(Termo legislativo) REL", "Relatório"),
            ("(Termo legislativo) REM", "Reclamação"),
            ("(Termo legislativo) REP", "Representação"),
            ("(Termo legislativo) REQ", "Requerimento"),
            ("(Termo legislativo) RIC", "Requerimento de Informação"),
            ("(Termo legislativo) RPR", "Representação"),
            ("(Termo legislativo) SBE", "Subemenda"),
            ("(Termo legislativo) SBT", "Substitutivo"),
            ("(Termo legislativo) SDL", "Sugestão de Emenda à LDO"),
            ("(Termo legislativo) SIT", "Solicitação de Informação ao TCU"),
            ("(Termo legislativo) SOA", "Sugestão de Emenda ao Orçamento"),
            (
                "(Termo legislativo) SUG",
                "Sugestão de entidade da sociedade civil à Câmara, para que adote alguma ação legislativa",
            ),
            ("(Termo jurídico) SUM", "Súmula de jurisprudência emitida pela CCJC"),
            ("(Termo legislativo) TER", "Termo de Implementação"),
            (
                "(Termo legislativo) TVR",
                "Ato do Poder Executivo que submete à Câmara a concessão de serviços de radiodifusão, sonora e de imagens",
            ),
            ("(Termo legislativo) VTS", "Voto em Separado"),
            ("MDB", "PARTIDO MOVIMENTO DEMOCRÁTICO BRASILEIRO"),
            ("PTB", "PARTIDO TRABALHISTA BRASILEIRO"),
            ("PDT", "PARTIDO DEMOCRÁTICO TRABALHISTA"),
            ("PT", "PARTIDO DOS TRABALHADORES"),
            ("PCdoB", "PARTIDO COMUNISTA DO BRASIL"),
            ("PSB", "PARTIDO SOCIALISTA BRASILEIRO"),
            ("PSDB", "PARTIDO DA SOCIAL DEMOCRACIA BRASILEIRA"),
            ("PSC", "PARTIDO SOCIAL CRISTÃO"),
            ("PMN", "PARTIDO DA MOBILIZAÇÃO NACIONAL"),
            ("CIDADANIA", "PARTIDO CIDADANIA"),
            ("PV", "PARTIDO VERDE"),
            ("PP", "PARTIDO PROGRESSISTAS"),
            ("PSTU", "PARTIDO SOCIALISTA DOS TRABALHADORES UNIFICADO"),
            ("PCB", "PARTIDO COMUNISTA BRASILEIRO"),
            ("PRTB", "PARTIDO RENOVADOR TRABALHISTA BRASILEIRO"),
            ("DC", "PARTIDO DEMOCRACIA CRISTÃ"),
            ("PCO", "PARTIDO DA CAUSA OPERÁRIA"),
            ("PODE", "PARTIDO PODEMOS"),
            ("REPUBLICANOS", "PARTIDO REPUBLICANOS"),
            ("PSOL", "PARTIDO SOCIALISMO E LIBERDADE"),
            ("PL", "PARTIDO LIBERAL"),
            ("PSD", "PARTIDO SOCIAL DEMOCRÁTICO"),
            ("PATRIOTA", "PARTIDO PATRIOTA"),
            ("NOVO", "PARTIDO PARTIDO NOVO"),
            ("REDE", "PARTIDO REDE SUSTENTABILIDADE"),
            ("PMB", "PARTIDO DA MULHER BRASILEIRA"),
            ("UP", "PARTIDO UNIDADE POPULAR"),
            ("VTS", "Voto em Separado"),
            ("MDB", "partido Movimento Democrático Brasileiro"),
            ("PTB", "partido Trabalhista Brasileiro"),
            ("PDT", "Partido Democrático Trabalhista"),
            ("PT", "Partido Dos Trabalhadores"),
            ("PCdoB", "Partido Comunista Do Brasil"),
            ("PSB", "Partido Socialista Brasileiro"),
            ("PSDB", "Partido Da Social Democracia Brasileira"),
            ("PSC", "Partido Social Cristão"),
            ("PMN", "Partido Da Mobilização Nacional"),
            ("UNIÃO", "PARTIDO UNIÃO BRASIL"),
            ("CIDADANIA", "Partido Cidadania"),
            ("PV", "Partido Verde"),
            ("AVANTE", "Partido Avante"),
            ("PP", "Partido Progressistas"),
            ("PSTU", "Partido Socialista Dos Trabalhadores Unificado"),
            ("PCB", "Partido Comunista Brasileiro"),
            ("PRTB", "Partido Renovador Trabalhista Brasileiro"),
            ("DC", "Partido Democracia Cristã"),
            ("PCO", "Partido Da Causa Operária"),
            ("PODE", "Partido Podemos"),
            ("REPUBLICANOS", "Partido Republicanos"),
            ("PSOL", "Partido Socialismo E Liberdade"),
            ("PL", "Partido Liberal"),
            ("PSD", "Partido Social Democrático"),
            ("PATRIOTA", "Partido Patriota"),
            ("SOLIDARIEDADE", "Partido Solidariedade"),
            ("NOVO", "Partido Novo"),
            ("REDE", "Partido Rede Sustentabilidade"),
            ("PMB", "Partido Da Mulher Brasileira"),
            ("UP", "Partido Unidade Popular"),
            ("UNIÃO", "Partido União Brasil"),
            ("ADIN", "Ação Direta de Inconstitucionalidade"),
            ("ADIn", "Ação Direta de Inconstitucionalidade"),
            ("ADI", "Ação Direta de Inconstitucionalidade"),
            ("Estado do Acre", "AC"),
            ("Estado de Alagoas", "AL"),
            ("Estado do Amapá", "AP"),
            ("Estado do Amazonas", "AM"),
            ("Estado da Bahia", "BA"),
            ("Estado de Ceará", "CE"),
            ("Distrito Federal", "DF"),
            ("Estado do Espírito Santo", "ES"),
            ("Estado de Goiás", "GO"),
            ("Estado de Maranhão", "MA"),
            ("Estado do Mato Grosso", "MT"),
            ("Estado do Mato Grosso do Sul", "MS"),
            ("Estado de Minas Gerais", "MG"),
            ("Estado do Pará", "PA"),
            ("Estado da Paraíba", "PB"),
            ("Estado do Paraná", "PR"),
            ("Estado de Pernambuco", "PE"),
            ("Estado do Piauí", "PI"),
            ("Estado do Rio de Janeiro", "RJ"),
            ("Estado do Rio Grande do Norte", "RN"),
            ("Estado do Rio Grande do Sul", "RS"),
            ("Estado de Rondônia", "RO"),
            ("Estado de Roraima", "RR"),
            ("Estado de Santa Catarina", "SC"),
            ("Estado de São Paulo", "SP"),
            ("Estado de Sergipe", "SE"),
            ("Estado de Tocantins", "TO"),
            ("Supremo Tribunal Federal", "STF"),
            ("Superior Tribunal de Justiça", "STJ"),
            ("Conselho da Justiça Federal", "CJF"),
            ("Superior Tribunal Militar", "STM"),
            ("Tribunal Superior do Trabalho", "TST"),
            ("Conselho Superior da Justiça do Trabalho", "CSJT"),
            ("Tribunal Superior Eleitoral", "TSE"),
            (
                "Tribunal Regional Federal da 1ª Região",
                "TRF1 (AC, AM, AP, BA, DF, GO, MA, MT, PA, PI, RO, RR e TO)",
            ),
            ("Tribunal Regional Federal da 2ª Região", "TRF2 (ES e RJ)"),
            ("Tribunal Regional Federal da 3ª Região", "TRF3 (MS e SP)"),
            ("Tribunal Regional Federal da 4ª Região", "TRF4 (PR, RS e SC)"),
            (
                "Tribunal Regional Federal da 5ª Região",
                "TRF5 (AL, CE, PB, PE, RN e SE)",
            ),
            ("Tribunal Regional Federal da 6ª Região", "TRF6 (MG)"),
            ("Tribunal de Justiça do Acre", "TJAC"),
            ("Tribunal de Justiça de Alagoas", "TJAL"),
            ("Tribunal de Justiça do Amapá", "TJAP"),
            ("Tribunal de Justiça do Amazonas", "TJAM"),
            ("Tribunal de Justiça da Bahia", "TJBA"),
            ("Tribunal de Justiça do Ceará", "TJCE"),
            ("Tribunal de Justiça do Distrito Federal e Territórios", "TJDFT"),
            ("Tribunal de Justiça do Espírito Santo", "TJES"),
            ("Tribunal de Justiça de Goiás", "TJGO"),
            ("Tribunal de Justiça do Maranhão", "TJMA"),
            ("Tribunal de Justiça de Mato Grosso", "TJMT"),
            ("Tribunal de Justiça de Mato Grosso do Sul", "TJMS"),
            ("Tribunal de Justiça de Minas Gerais", "TJMG"),
            ("Tribunal de Justiça do Pará", "TJPA"),
            ("Tribunal de Justiça da Paraíba", "TJPB"),
            ("Tribunal de Justiça do Paraná", "TJPR"),
            ("Tribunal de Justiça de Pernambuco", "TJPE"),
            ("Tribunal de Justiça do Piauí", "TJPI"),
            ("Tribunal de Justiça do Rio de Janeiro", "TJRJ"),
            ("Tribunal de Justiça do Rio Grande do Norte", "TJRN"),
            ("Tribunal de Justiça do Rio Grande do Sul", "TJRS"),
            ("Tribunal de Justiça de Rondônia", "TJRO"),
            ("Tribunal de Justiça de Roraima", "TJRR"),
            ("Tribunal de Justiça de Santa Catarina", "TJSC"),
            ("Tribunal de Justiça de São Paulo", "TJSP"),
            ("Tribunal de Justiça de Sergipe", "TJSE"),
            ("Tribunal de Justiça do Tocantins", "TJTO"),
            ("Tribunal Regional do Trabalho da 1ª Região", "TRT1 (RJ)"),
            (
                "Tribunal Regional do Trabalho da 2ª Região",
                "TRT2 (SP / Grande São Paulo e Baixada Santista)",
            ),
            ("Tribunal Regional do Trabalho da 3ª Região", "TRT3 (MG)"),
            ("Tribunal Regional do Trabalho da 4ª Região", "TRT4 (RS)"),
            ("Tribunal Regional do Trabalho da 5ª Região", "TRT5 (BA)"),
            ("Tribunal Regional do Trabalho da 6ª Região", "TRT6 (PE)"),
            ("Tribunal Regional do Trabalho da 7ª Região", "TRT7 (CE)"),
            ("Tribunal Regional do Trabalho da 8ª Região", "TRT8 (AP e PA)"),
            ("Tribunal Regional do Trabalho da 9ª Região", "TRT9 (PR)"),
            ("Tribunal Regional do Trabalho da 10ª Região", "TRT10 (DF e TO)"),
            ("Tribunal Regional do Trabalho da 11ª Região", "TRT11 (AM e RR)"),
            ("Tribunal Regional do Trabalho da 12ª Região", "TRT12 (SC)"),
            ("Tribunal Regional do Trabalho da 13ª Região", "TRT13 (PB)"),
            ("Tribunal Regional do Trabalho da 14ª Região", "TRT14 (AC e RO)"),
            (
                "Tribunal Regional do Trabalho da 15ª Região",
                "TRT15 (SP / Interior e Litoral Norte e Sul)",
            ),
            ("Tribunal Regional do Trabalho da 16ª Região", "TRT16 (MA)"),
            ("Tribunal Regional do Trabalho da 17ª Região", "TRT17 (ES)"),
            ("Tribunal Regional do Trabalho da 18ª Região", "TRT18 (GO)"),
            ("Tribunal Regional do Trabalho da 19ª Região", "TRT19 (AL)"),
            ("Tribunal Regional do Trabalho da 20ª Região", "TRT20 (SE)"),
            ("Tribunal Regional do Trabalho da 21ª Região", "TRT21 (RN)"),
            ("Tribunal Regional do Trabalho da 22ª Região", "TRT22 (PI)"),
            ("Tribunal Regional do Trabalho da 23ª Região", "TRT23 (MT)"),
            ("Tribunal Regional do Trabalho da 24ª Região", "TRT24 (MS)"),
            ("Tribunal de Justiça Militar de Minas Gerais", "TJMMG"),
            ("Tribunal de Justiça Militar do Rio Grande do Sul", "TJMRS "),
            ("Tribunal de Justiça Militar de São Paulo", "TJMSP"),
            ("BC", "Banco Central do Brasil"),
            ("BB", "Banco do Brasil"),
            (
                "(Termo jurídico) AÇÃO",
                "instrumento para o cidadão reivindicar ou defender um direito na Justiça; exercício do direito de acesso ao Tribunal.",
            ),
            (
                "(Termo jurídico) AÇÃO ORIGINÁRIA",
                "ação que, em função da matéria ou das partes, é processada desde o início no TRF.",
            ),
            (
                "(Termo jurídico) AÇÃO PENAL",
                "ação em que se apura e julga a ocorrência de um crime ou de uma contravenção.",
            ),
            (
                "(Termo jurídico) AÇÃO PENAL PÚBLICA",
                "ação Penal de iniciativa do Ministério Público.",
            ),
            (
                "(Termo jurídico) AÇÃO RESCISÓRIA",
                "ação que pede a anulação de uma sentença transitada em julgado (de que não cabe mais recurso) considerada ilegal.",
            ),
            (
                "(Termo jurídico) ACÓRDÃO",
                "decisão de Turma, Seção ou Plenário do Tribunal.",
            ),
            (
                "(Termo jurídico) ADITAMENTO EM PAUTA",
                "complementação da pauta de julgamento, a fim de incluir novos processos, ou para que sejam feitas correções.",
            ),
            (
                "(Termo jurídico) AGU – ADVOCACIA GERAL DA UNIÃO",
                "instituição que representa os interesses da União em questões judiciais e extrajudiciais. Presta ainda assessoria jurídica e consultoria ao Poder Executivo da União. Os membros da carreira são advogados da União, procuradores da Fazenda Nacional e assistentes jurídicos. O chefe da instituição é o advogado-geral da União.",
            ),
            (
                "(Termo jurídico) AGRAVO",
                "recurso contra decisão de juiz ou desembargador proferida no transcorrer do processo (interlocutória). É diferente da apelação contra a sentença ou decisão final do juiz ou do tribunal.",
            ),
            (
                "(Termo jurídico) AGRAVO DE INSTRUMENTO",
                "recurso apresentado diretamente ao Tribunal contra decisão interlocutória de um juiz de primeiro grau.",
            ),
            (
                "(Termo jurídico) AGUARDANDO PUBLICAÇÃO",
                "quando uma decisão, um despacho ou um acórdão já constam do processo, mas ainda não foram publicados no Diário da Justiça, após o que passarão a produzir seus efeitos.",
            ),
            (
                "(Termo jurídico) ALVARÁ",
                "documento judicial expedido para autorizar o levantamento de quantias (alvará de levantamento), para liberação de preso (alvará de soltura), para o funcionamento de uma empresa (alvará de funcionamento).",
            ),
            ("(Termo jurídico) APENSADO", "quando um processo é anexado a outro."),
            (
                "(Termo jurídico) ARGUIÇÃO DE SUSPEIÇÃO",
                "processo para afastar do caso um juiz, membro do Ministério Público ou servidor da Justiça que se desconfie de ser parcial em um caso, por ter motivo para estar interessado nele.",
            ),
            (
                "(Termo jurídico) BAIXA",
                "quando um processo é remetido à Vara de origem ou a outro Órgão de Primeira Instância.",
            ),
            (
                "(Termo jurídico) CARTA PRECATÓRIA",
                "documento judicial solicitando diligencia (providencia) a juiz de outra comarca.",
            ),
            (
                "(Termo jurídico) CARTA ROGATÓRIA",
                "pedido feito por autoridade judicial estrangeira para que seja cumprida uma diligência no Brasil, como citação, interrogatório de testemunhas, prestação de informações, entre outras.",
            ),
            (
                "(Termo jurídico) CONCLUSÃO",
                "quando o processo está com o Juiz ou Desembargador Relator para redigir a decisão, o acórdão ou um despacho (para que decida sobre determinada questão e providências a serem adotadas).",
            ),
            (
                "(Termo jurídico) CONCLUSÃO A(O) VICE-PRESIDENTE PARA EXAME DE ADMISSIBILIDADE",
                "quando o processo está no Gabinete do Vice-Presidente para admitir ou não um recurso para o Supremo Tribunal Federal e/ou para o Superior Tribunal de Justiça (em Brasília).",
            ),
            (
                "(Termo jurídico) CONFLITO DE COMPETÊNCIA",
                "ação para decidir qual autoridade judiciária tem poder para atuar em determinada situação. A ação pode ser proposta pela parte interessada, pelo Ministério Público ou por uma das autoridades em conflito.",
            ),
            (
                "(Termo jurídico) CONTRA-RAZÕES",
                "manifestação (defesa) da parte, contra a qual foi apresentado um recurso.",
            ),
            (
                "(Termo jurídico) CORREGEDORIA GERAL",
                "órgão de fiscalização, disciplina e orientação administrativa do Tribunal.",
            ),
            (
                "(Termo jurídico) CORREIÇÃO PARCIAL",
                "recurso que visa à emenda de erros ou abusos que importem a inversão tumultuária de atos e fórmulas legais, a paralisação injustificada dos feitos ou a dilatação abusiva dos prazos por parte dos Juízes da Turma no Tribunal ou dos juízes de primeiro grau, quando, para o caso, não haja recurso previsto em lei.",
            ),
            (
                "(Termo jurídico) DECISÃO DE ADMISSIBILIDADE",
                "decisão que admite ou não um recurso.",
            ),
            (
                "(Termo jurídico) DECISÃO DEFINITIVA",
                "decisão final em um processo. Pode ser uma sentença, quando é tomada por um juiz, ou acórdão, quando é proferida pelo tribunal.",
            ),
            (
                "(Termo jurídico) DECISÃO MONOCRÁTICA",
                "decisão do Desembargador Relator que põe fim à demanda, sem submeter o processo à Turma para julgamento (vide: artigo 557 do Código de Processo Civil).",
            ),
            (
                "(Termo jurídico) DENÚNCIA",
                "ato pelo qual o membro do Ministério Público (promotor ou procurador da República) formaliza a acusação perante o Poder Judiciário, dando início à ação penal. Só cabe em ação pública (na ação privada, existe a Queixa). Se a denúncia for recebida pelo juiz (ou, no Tribunal, pela Seção – reunião de turmas de matéria penal) o denunciado passa a ser réu na ação penal.",
            ),
            (
                "(Termo jurídico) DESPACHO",
                "decisão através da qual o Juiz determina alguma providência necessária para o andamento do processo. São chamados “de mero expediente” quando não tem caráter decisório, servindo, apenas, para movimentar o processo (por exemplo, para pedir que se ouçam as partes).",
            ),
            (
                "(Termo jurídico) DIA PARA JULGAMENTO",
                "quando o processo está na Turma aguardando definição de data para julgamento.",
            ),
            (
                "(Termo jurídico) DIÁRIO DA JUSTIÇA",
                "Diário Oficial onde são publicadas as decisões do Poder Judiciário.",
            ),
            (
                "(Termo jurídico) DILIGÊNCIA",
                "providência determinada pelo juiz ou desembargador para esclarecer alguma questão do processo. Pode ser decidida por iniciativa do juiz (de ofício) ou atendendo a requerimento do Ministério Público ou das partes.",
            ),
            (
                "(Termo jurídico) DISTRIBUIÇÃO AUTOMÁTICA",
                "a ação (ou o recurso) logo que chega ao Tribunal é distribuída, através de sorteio eletrônico, para um dos Desembargadores (ou Juízes Convocados), que ficará como o Relator do processo e tomará as providências necessárias para que seja julgado. Pode acontecer também por prevenção, ou seja, o processo é distribuído para um juiz ou desembargador que já tenha atuado em causa ou processo conexo.",
            ),
            (
                "(Termo jurídico) DISTRIBUIÇÃO COM RECURSO EXTRA-ORDINÁRIO E/OU ESPECIAL",
                "o processo será distribuído (vide: DISTRIBUIÇÃO AUTOMÁTICA), contendo recursos para o Supremo Tribunal Federal e/ou Superior Tribunal de Justiça (em Brasília).",
            ),
            (
                "(Termo jurídico) EFEITO SUSPENSIVO",
                "suspensão dos efeitos da decisão de um juiz ou tribunal, até que a instância superior tome a decisão final sobre o recurso interposto.",
            ),
            (
                "(Termo jurídico) EM MESA PARA JULGAMENTO (JULGADO EM MESA)",
                "quando o processo ou o incidente processual é levado à Turma para julgamento independente de inclusão prévia na pauta, sem ter sido incluído previamente na “ordem do dia”.",
            ),
            (
                "(Termo jurídico) EMBARGOS",
                "espécie de recurso ordinário para determinado provimento judicial. Os mais comuns são os embargos declaratórios. No TRF, também cabem os embargos infringentes.",
            ),
            (
                "(Termo jurídico) EMBARGOS DECLARATÓRIOS",
                "embargos que pedem que se esclareça um ponto da decisão judicial considerado obscuro, contraditório, omisso ou duvidoso.",
            ),
            (
                "(Termo jurídico) EMBARGOS INFRINGENTES",
                "recurso cabível de julgamento não unânime proferido em apelação, remessa ex officio e em ação rescisória. A impugnação deve recair somente sobre a matéria objeto de divergência.",
            ),
            ("(Termo jurídico) EMENTA", "resumo de uma decisão judiciária."),
            (
                "(Termo jurídico) EX NUNC",
                "expressão latina. Quer dizer que a decisão não tem efeito retroativo, ou seja, vale do momento em que foi proferida em diante.",
            ),
            (
                "(Termo jurídico) EX TUNC",
                "expressão latina. Quer dizer que a decisão tem efeito retroativo, valendo também para o passado.",
            ),
            (
                "(Termo jurídico) FUMUS BONI IURIS",
                "expressão latina. Quer dizer “fumaça do bom direito”, quando o juiz decide baseado na presunção de grande probabilidade da existência do direito no caso concreto.",
            ),
            (
                "(Termo jurídico) HABEAS CORPUS",
                "ação que visa a proteger o direito de ir e vir. É concedido sempre que alguém sofrer ou se achar ameaçado de sofrer violência ou coação em sua liberdade de locomoção, por ilegalidade ou abuso de poder. Quando há apenas ameaça ao direito acima exposto, o habeas corpus é preventivo.",
            ),
            (
                "(Termo jurídico) HABEAS DATA",
                "ação para garantir o acesso de uma pessoa a informações sobre ela que façam parte de arquivos ou bancos de dados de entidades governamentais ou públicas. Também pode pedir a correção de dados incorretos.",
            ),
            (
                "(Termo jurídico) IMPEDIMENTO",
                "situação em que um juiz é proibido de atuar num processo. Pode dar-se por declaração do próprio magistrado.",
            ),
            (
                "(Termo jurídico) IMPRENSA NACIONAL",
                "Órgão Público responsável pela publicação dos atos e decisões dos Poderes Executivo, Judiciário e Legislativo.",
            ),
            ("(Termo jurídico) IMPUGNAR", "contestar."),
            (
                "(Termo jurídico) INCIDENTE",
                "questão relevante que deve ser previamente examinada.",
            ),
            (
                "(Termo jurídico) INCLUÍDO EM PAUTA",
                "quando é marcada a data para o julgamento do processo ou incidente.",
            ),
            (
                "(Termo jurídico) INQUÉRITO",
                "procedimento para apurar a ocorrência de infração penal. A partir do Inquérito se reúnem elementos para o Ministério Público decidir se denuncia ou não o acusado perante o Poder Judiciário.",
            ),
            ("(Termo jurídico) INTEMPESTIVO", "fora do prazo."),
            (
                "(Termo jurídico) INTIMAÇÃO",
                "dar ciência do teor de decisão ou de acórdão.",
            ),
            (
                "(Termo jurídico) INTIMAÇÃO PARA CONTRA-RAZÕES",
                "quando é dada ciência às partes para que apresentem seus argumentos (sua defesa) contra um recurso.",
            ),
            (
                "(Termo jurídico) INSTÂNCIA",
                "grau da hierarquia do Poder Judiciário. A primeira instância, onde em geral começam as ações, é composta pelo Juízo de Direito de cada comarca, pelo Juízo Federal, Eleitoral e do Trabalho. A segunda instância, onde são julgados recursos, é formada pelos Tribunais de Justiça e pelos Tribunais Regionais Federais, Eleitorais e do Trabalho. A terceira instância são os Tribunais Superiores (STF, STJ, TST, TSE) que julgam recursos contra decisões dos tribunais de segunda instância.",
            ),
            (
                "(Termo jurídico) INTERESSE DIFUSO",
                "interesse em relação a questões que dizem respeito a toda coletividade, de forma indeterminada. Por exemplo, habitação e saúde.",
            ),
            (
                "(Termo jurídico) JULGADO POR DECISÃO MONOCRÁTICA",
                "quando o processo é julgado no Gabinete por meio de uma decisão do Desembargador ou Juiz Relator, sem que seja levado à Turma para julgamento.",
            ),
            (
                "(Termo jurídico) JUNTADO(A)",
                "quando é anexado algum documento ao processo.",
            ),
            (
                "(Termo jurídico) JURISPRUDÊNCIA",
                "conjunto de decisões do tribunal num mesmo sentido.",
            ),
            (
                "(Termo jurídico) JUSTIÇA FEDERAL",
                "órgão do Poder Judiciário composto pelos Tribunais Regionais Federais e pelos juízes federais.",
            ),
            (
                "(Termo jurídico) LEI",
                "regra geral e permanente a que todos estão submetidos.",
            ),
            (
                "(Termo jurídico) LIMINAR",
                "ordem judicial que garante a antecipação de um direito. É concedida quando a demora da decisão puder causar prejuízos irreparáveis ou de difícil reparação. Ao examinar a liminar, o juiz ou desembargador relator também avalia se o pedido apresentado tem fundamentos jurídicos aceitáveis.",
            ),
            (
                "(Termo jurídico) LOCALIZAÇÃO",
                "local onde está o processo (Ex: na Turma, no Gabinete do Desembargador, etc.)",
            ),
            (
                "(Termo jurídico) LITISCONSÓRCIO",
                "concomitância de mais de uma pessoa na posição de autor ou de réu, no mesmo processo.",
            ),
            (
                "(Termo jurídico) MANDADO",
                "ordem escrita da autoridade. É chamado de mandado judicial quando expedido por juiz ou desembargador de Tribunal. Tem nomes específicos de acordo com o objetivo: prisão, soltura, etc.",
            ),
            (
                "(Termo jurídico) MANDADO DE SEGURANÇA",
                "ação para garantir direito líquido e certo, individual ou coletivo, que esteja sendo violado ou ameaçado por ato de uma autoridade, em ato ilegal ou inconstitucional.",
            ),
            (
                "(Termo jurídico) MEDIDA CAUTELAR",
                "ação destinada a garantir a efetividade da futura execução da prestação pleiteada em um processo de conhecimento. Os requisitos para sua concessão são a probabilidade de êxito na ação principal (fumus boni iuris) e o risco de a prestação pretendida ser frustrada (periculum in mora)",
            ),
            (
                "MPF – Ministério Público Federal",
                "instituição essencial ao funcionamento da Justiça na Constituição de 1988 (Arts. 127 a 130). Seus objetivos são fiscalizar o cumprimento da lei, defender a democracia e os direitos individuais, coletivos e difusos. Os membros do Ministério Público dos estados e do Distrito Federal são promotores e procuradores de Justiça. Os membros do Ministério Público Militar são promotores e procuradores de Justiça Militar. Os membros do Ministério Público do Trabalho são procuradores do Trabalho. Os membros do Ministério Público Federal são procuradores da República.",
            ),
            (
                "(Termo jurídico) PARECER",
                "opinião técnica de advogado, consultor jurídico, membro do Ministério Público ou qualquer funcionário competente sobre determinado assunto. Juízes decidem ou despacham, não dão pareceres.",
            ),
            (
                "(Termo jurídico) PARTE",
                "toda pessoa que participa de um processo. Pode ser a parte que provocou o processo, autor, ou a parte que se defende, réu.",
            ),
            (
                "(Termo jurídico) PAUTA DE JULGAMENTO",
                "relação de processos que serão julgados em determinado dia.",
            ),
            (
                "(Termo jurídico) PEDIDO DE VISTA",
                "quando um Desembargador (ou Juiz Convocado) solicita o processo para exame.",
            ),
            (
                "(Termo jurídico) PERICULUM IM MORA",
                "expressão latina. Quer dizer “perigo na demora”, significando que o pedido deve ser analisado com urgência, para evitar dano grave e de difícil reparação.",
            ),
            (
                "(Termo jurídico) PETIÇÃO",
                "de forma geral, é um pedido escrito dirigido ao Tribunal, feito através de advogado. A Petição Inicial é o pedido para que se comece um processo. Outras petições podem ser apresentadas durante o processo para requerer o que é de interesse ou de direito das partes.",
            ),
            ("(Termo jurídico) PETIÇÃO DESENTRANHADA", "petição retirada do processo."),
            (
                "(Termo jurídico) PRECATÓRIO",
                "determinação da Justiça para que um órgão público (governo estadual, fundação, etc.) pague uma indenização devida. Os precatórios devem ser pagos em ordem cronológica, quer dizer, primeiro os mais antigos, independentemente do valor.",
            ),
            (
                "(Termo jurídico) PRECATÓRIO LIQUIDADO",
                "diz-se do precatório quando a quantia devida já foi paga.",
            ),
            ("(Termo jurídico) PREPARO", "custas judiciais relativas a recursos."),
            (
                "(Termo jurídico) PRIMEIRA INSTÂNCIA",
                "diz-se da Justiça Federal de Primeiro Grau (Varas Federais), onde o processo originário será julgado por um Juiz Federal.",
            ),
            (
                "(Termo jurídico) PRISÃO PREVENTIVA",
                "medida restritiva da liberdade decretada antes de decisão judicial transitada em julgado. Essa segregação tem por objetivo acautelar a ordem pública ou econômica, evitar que o réu se exima da aplicação da lei penal, ou propiciar o adequado andamento da instrução criminal (impedindo, por exemplo, que o réu destrua provas ou influencie testemunhas).",
            ),
            (
                "(Termo jurídico) PROCURADOR FEDERAL",
                "representante de órgãos da administração indireta da União – autarquias e de fundações – em questões judiciais e extrajudiciais.",
            ),
            (
                "(Termo jurídico) PROCURADOR GERAL DA REPÚBLICA",
                "chefe do Ministério Público Federal e do Ministério Público da União. É escolhido pelo presidente da República, entre os integrantes da carreira maiores de 35 anos, e aprovado pelo Senado Federal. Tem mandato de dois anos, permitidas reconduções. Sua destituição, pelo presidente da República, depende de autorização do Senado. O procurador-geral da República é processado e julgado pelo STF.",
            ),
            (
                "(Termo jurídico) QUEIXA",
                "exposição do fato criminoso, feita pela parte ofendida ou por seu representante legal, para iniciar processo contra o autor ou autores do crime processado por meio de ação penal privada. A queixa pode ser apresentada por qualquer cidadão – é um procedimento penal de caráter privado, que corresponde à Denúncia na ação penal pública. A queixa não está sujeita a formalidades especiais, podendo ser feita oralmente (Lei 9099/95) ou por escrito. O prazo de apresentação da queixa é de seis meses, a contar da data em que o denunciante tomou conhecimento do crime e dos seus autores.",
            ),
            (
                "(Termo jurídico) QUORUM",
                "número mínimo de desembargadores necessário para os julgamentos.",
            ),
            (
                "(Termo jurídico) RAZÕES",
                "argumentos e fatos alegados pela parte com o objetivo de modificar a decisão do Juiz.",
            ),
            (
                "(Termo jurídico) RECURSO",
                "instrumento para pedir a mudança de uma decisão, na mesma instância ou em instância superior. Existem vários tipos de recursos: embargo, agravo, apelação, recurso especial, recurso extraordinário, etc.",
            ),
            (
                "(Termo jurídico) RECURSO ESPECIAL",
                "recurso ao Superior Tribunal de Justiça, de caráter excepcional, contra decisões de outros tribunais, em única ou última instância, quando houver ofensa à lei federal. Também é usado para pacificar a jurisprudência, ou seja, para unificar interpretações divergentes feitas por diferentes tribunais sobre o mesmo assunto.",
            ),
            (
                "(Termo jurídico) RECURSO EXTRAORDINÁRIO",
                "recurso de caráter excepcional para o Supremo Tribunal Federal contra decisões de outros tribunais, em única ou última instância, quando houver ofensa a norma da Constituição Federal.",
            ),
            (
                "(Termo jurídico) RECURSO JULGADO DESERTO",
                "diz-se do recurso que é negado por falta de pagamento das custas judiciais.",
            ),
            (
                "(Termo jurídico) REDISTRIBUIÇÃO",
                "em alguns casos o processo é distribuído novamente, sendo designado um novo Relator.",
            ),
            (
                "(Termo jurídico) RELATOR",
                "desembargador sorteado para dirigir um processo. Também pode ser escolhido por prevenção, quando já for o relator de processo relativo ao mesmo assunto. O relator decide ou, conforme o caso, leva seu voto para decisão pela turma ou pelo plenário.",
            ),
            (
                "(Termo jurídico) REMESSA AO ARQUIVO (ARQUIVADO)",
                "quando o processo, para sua guarda e conservação, é enviado para o Arquivo do Tribunal (Rua Acre 80 – sala 505 – Prédio Anexo II B).",
            ),
            (
                "(Termo jurídico) REMESSA COM BAIXA (NA DISTRIBUIÇÃO)",
                "quando o processo é encaminhado (“transferido”) a outros Órgãos Externos ou ao Arquivo, cancelando-se o seu registro de entrada no Tribunal.",
            ),
            (
                "(Termo jurídico) REMESSA EX-OFFICIO",
                "processo que sobe ao Tribunal em cumprimento da exigência do duplo grau de jurisdição independentemente da manifestação recursal.",
            ),
            (
                "(Termo jurídico) SEGUNDA INSTÂNCIA",
                "diz-se da Justiça Federal de Segundo Grau (Tribunais), onde os recursos serão julgados por Desembargadores.",
            ),
            ("(Termo jurídico) SENTENÇA", "decisão do Juiz que põe fim à demanda."),
            (
                "(Termo jurídico) REVISÃO CRIMINAL",
                "pedido do condenado para que a sentença seja reexaminada, argumentando que ela é incorreta, em casos previstos na lei. A Revisão criminal é ajuizada quando já não cabe nenhum outro recurso contra a decisão.",
            ),
            (
                "(Termo jurídico) REVISOR",
                "desembargador que a quem incumbe revisar o processo penal, após o relatório do desembargador-relator.",
            ),
            (
                "(Termo jurídico) SEÇÃO",
                "órgão fracionário do Tribunal, formado pela reunião dos componentes de duas Turmas julgadoras da mesma matéria.",
            ),
            ("(Termo jurídico) SENTENÇA", "decisão do juiz que põe fim a um processo."),
            (
                "(Termo jurídico) SÚMULA",
                "registro da jurisprudência dominante do Tribunal.",
            ),
            (
                "(Termo jurídico) TRANSITADA EM JULGADO",
                "expressão usada para uma decisão (sentença ou acórdão) de que não se pode mais recorrer, seja porque já passou por todos os recursos possíveis, seja porque o prazo para recorrer terminou.",
            ),
            (
                "(Termo jurídico) TURMA",
                "Órgão Julgador, composto por, no mínimo, três Desembargadores (ou Juízes Convocados), que, em conjunto, julgam os processos no Tribunal.",
            ),
            (
                "(Termo jurídico) TUTELA ANTECIPADA (ANTECIPAÇÃO DE TUTELA)",
                "antecipação de um direito antes da decisão final do processo.",
            ),
            (
                "(Termo jurídico) VARA DE ORIGEM",
                "Vara Federal na qual foi julgado o processo originário.",
            ),
            (
                "(Termo jurídico) VISTA",
                "retirada do processo para análise, pela parte, através de seu advogado, pelo Ministério Público, pelo Perito, entre outros.",
            ),
        ]
    )

    pairs.extend(
        [
            (
                "(MPF) Notícia de Fato (NF)",
                "Qualquer demanda dirigida aos órgãos da atividade-fim do MPF, submetida à apreciação das Procuradorias, que ainda não tenha gerado um feito interno ou externo, entendendo-se como tal, a entrada de atendimentos, noticias, documentos ou representações",
            ),
            (
                "(MPF) Procedimento Preparatório (PP)",
                "Procedimento formal, prévio ao Inquérito Civil, que visa apurar elementos para identificação dos investigados ou do objeto",
            ),
            (
                "(MPF) Procedimento Administrativo (PA)",
                "Procedimento destinado ao acompanhamento de fiscalizações, de cunho permanente ou não, de fatos e instituições e de políticas públicas e demais procedimentos não sujeitos a inquérito civil que não tenham o caráter de investigação cível ou criminal de determinada pessoa, em função de um ilícito específico",
            ),
            (
                "(MPF) Inquérito Civil (IC)",
                "Procedimento de natureza administrativa, instaurado mediante portaria, onde são reunidos oficialmente os documentos produzidos no decurso de uma investigação destinada a constatar desrespeito a direitos constitucionalmente assegurados ao cidadão, dano ao patrimônio público ou social ou a direitos difusos, coletivos e individuais indisponíveis",
            ),
            (
                "(MPF) Procedimento Investigatório Criminal (PIC)",
                "Instrumento de coleta de dados, destinado a apurar a ocorrência de infrações penais de natureza pública, servindo como preparação e embasamento para o juízo de propositura, ou não, da ação penal respectiva",
            ),
            ("(MPF) PR", "Procuradoria da República nos Estados"),
            ("(MPF) PRR", "Procuradoria Regional da República"),
            ("(MPF) PGR", "Procuradoria Geral da República"),
        ]
    )

    pairs.extend(
        [
            (
                "Controle do preço dos combustíveis",
                "Fundamentado na Lei nº 9.478/1997, desde janeiro de 2002 vigora no Brasil o regime de liberdade de preços em todos os segmentos do mercado de combustíveis e derivados de petróleo: produção, distribuição e revenda. Isso significa que não há qualquer tipo de tabelamento nem fixação de valores máximos e mínimos, ou qualquer exigência de autorização oficial prévia para reajustes.",
            ),
            (
                "Controle do preço dos combustíveis",
                "A Lei do Petróleo também criou a Agência Nacional do Petróleo - ANP e conferiu-lhe a competência para implementar a política energética nacional no que se refere a petróleo, gás natural e biocombustíveis, com ênfase na garantia do suprimento de derivados de petróleo, de gás natural e seus derivados e de biocombustíveis em todo o território nacional, e na proteção dos interesses do consumidor quanto a preço, qualidade e oferta desses produtos.",
            ),
            (
                "Linhas de financiamentos para projetos de energia",
                "O Ministério de Minas e Energia – MME, órgão da administração federal direta, representa a União como Poder Concedente e é formulador de políticas públicas, bem como indutor e supervisor da implementação dessas políticas no segmento de energia elétrica.",
            ),
            (
                "Tributos cobrados na conta de energia",
                "A tarifa de energia elétrica deve garantir o fornecimento de energia com qualidade e assegurar aos prestadores dos serviços receitas suficientes para cobrir custos operacionais eficientes e remunerar investimentos necessários para expandir a capacidade e garantir o atendimento.",
            ),
            (
                "Horário de Verão",
                "O Presidente da República, no uso da atribuição que lhe confere o art. 84, caput, inciso IV, da Constituição, decretou a suspensão do Horário Brasileiro de Verão. A cerimônia de assinatura do Decreto 9.772, de 25 de abril de 2019, aconteceu no Palácio do Planalto e contou com a presença do Ministro de Minas e Energia.",
            ),
            (
                "Horário de Verão",
                "A suspensão foi resultado dos estudos que comprovaram a neutralidade econômica da medida no âmbito do setor elétrico. Atualmente, a medida deixou de produzir economia como nos anos anteriores em razão das mudanças no hábito de consumo de energia da população.",
            ),
        ]
    )

    pairs.extend(
        [
            ("Leis de Sistema de Cultura - Ceará", "(CE) Lei nº 13.811, de 16/08/2006"),
            ("Leis de Sistema de Cultura - Acre", "(AC) Lei nº 2.312, de 15/10/2010"),
            ("Leis de Sistema de Cultura - Bahia", "(BA) Lei nº 12.365, de 30/11/2011"),
            ("Leis de Sistema de Cultura - Rondônia", "(RO) Lei nº  2.746, de 18/05/2012"),
            ("Leis de Sistema de Cultura - Rio Grande do Sul", "(RS) Lei nº 14.310, de 30/09/2013"),
            ("Leis de Sistema de Cultura - Paraíba", "(PB) Lei nº 10.325, de 11/06/2014"),
            ("Leis de Sistema de Cultura - Sergipe", "(SE) Lei nº 8.005, de 12/05/2015"),
            ("Leis de Sistema de Cultura - Rio de Janeiro", "(RJ) Lei nº 7.035, de 07/07/2015"),
            ("Leis de Sistema de Cultura - Mato Grosso", "(MG) Lei nº  10.363, de 27/01/2016"),
            ("Leis de Sistema de Cultura - Roraima", "(RR) Lei nº 1.033, de 22/03/2016"),
            ("Leis de Sistema de Cultura - Amapá", "(AM) Lei nº 2.137, de 02/03/2017"),
            ("Leis de Sistema de Cultura - Mato Grosso do Sul", "(MS) Lei nº 5.060, de 20/09/2017"),
            (
                "Leis de Sistema de Cultura - Distrito Federal",
                "(DF) Lei Comp. nº 934, de 7/12/2017",
            ),
            ("Leis de Sistema de Cultura - Santa Catarina", "(SC) Lei nº 17.449, de 10/01/2018"),
            ("Leis de Sistema de Cultura - Minas Gerais", "(MG) Lei nº 22.944, de 15/01/2018"),
            (
                "Criado em 1991 pela Lei 8.313, o mecanismo do incentivo à cultura é um dos pilares do Programa Nacional de Apoio à Cultura (Pronac), que também conta com o Fundo Nacional de Cultura (FNC) e os Fundos de Investimento Cultural e Artístico (Ficarts).",
                "Principal ferramenta de fomento à Cultura do Brasil, a Lei de Incentivo à Cultura contribui para que milhares de projetos culturais aconteçam, todos os anos, em todas as regiões do país.",
            ),
        ]
    )

    assert len(pairs)

    return pairs
