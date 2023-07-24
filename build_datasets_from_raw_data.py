import os
import argparse

import pandas as pd

import dataset_builder


def save_cache(dfs: list[pd.DataFrame], output_dir: str = "./processed_data") -> None:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output_dir, "positive_pairs.tsv"), sep="\t")


def main(args) -> None:
    os.makedirs(args.cache_dir, exist_ok=True)

    dfs = []

    pairs = dataset_builder.general.make_pairs_tv_camara(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="tv_camara"))
    print(dfs[-1])
    exit(0)

    pairs = dataset_builder.general.make_pairs_mpt(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_mpt"))

    pairs = dataset_builder.general.make_pairs_mpm(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_mpm"))

    pairs = dataset_builder.general.make_pairs_tcu(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_tcu"))

    pairs = dataset_builder.general.make_pairs_radio_camara(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="radio_camara"))

    pairs = dataset_builder.general.make_pairs_trf4(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf4"))

    pairs = dataset_builder.general.make_pairs_trf1(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf1"))

    pairs = dataset_builder.general.make_pairs_stj(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_stj"))

    pairs = dataset_builder.general.make_pairs_bc(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_bc"))

    pairs = dataset_builder.general.make_pairs_camara_noticias(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_camara"))

    pairs = dataset_builder.general.make_pairs_senado_noticias(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_senado"))

    pairs = dataset_builder.general.make_pairs_stf(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_stf"))

    pairs = dataset_builder.general.make_pairs_stm(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_stm"))

    pairs = dataset_builder.general.make_pairs_tst(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_tst"))

    pairs = dataset_builder.general.make_pairs_tst_radio(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="radio_tst"))

    pairs = dataset_builder.general.make_pairs_tst_tv(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="tv_tst"))

    pairs = dataset_builder.general.make_pairs_trf6(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf6"))

    pairs = dataset_builder.general.make_pairs_trf5(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf5"))

    pairs = dataset_builder.general.make_pairs_onu(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_onu"))

    pairs = dataset_builder.general.make_pairs_capes(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_capes"))

    pairs = dataset_builder.general.make_pairs_camara_noticias_comments()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_camara_comments"))

    pairs = dataset_builder.faqs.make_pairs_bc()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_bc"))

    pairs = dataset_builder.general.make_pairs_cnmp(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_cnmp"))

    pairs = dataset_builder.general.make_pairs_tse(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_tse"))

    pairs = dataset_builder.general.make_pairs_trf2(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf2"))

    pairs = dataset_builder.general.make_pairs_trf3(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="news_trf3"))

    pairs = dataset_builder.general.make_pairs_radio_e_tv_justica()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="radio_e_tv_justica"))

    pairs_dict = dataset_builder.general.make_pairs_ministerios(long_segments=args.long_segments)
    for tag, pairs in pairs_dict.items():
        dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name=f"news_min_{tag}"))
        assert len(dfs[-1]) >= 2, dfs[-1]

    pairs = dataset_builder.stf_annotated_laws.make_pairs_const()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_const"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_lei_9868_1999()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_lei_9868_1999"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_lei_9882_1999()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_lei_9882_1999"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_oab()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_oab"))

    pairs = dataset_builder.speeches.make_pairs(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="speeches"))

    pairs = dataset_builder.mentions.make_pairs()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="law_mentions"))

    pairs = dataset_builder.chatgpt_data.make_pairs(task="map2doc")
    dfs.append(
        dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="chatgpt_map2doc")
    )

    pairs = dataset_builder.chatgpt_data.make_pairs(task="clusterComments")
    dfs.append(
        dataset_builder.utils.gen_dataframe(
            pairs, min_length=1, source_name="chatgpt_clusterComments"
        )
    )

    pairs = dataset_builder.chatgpt_data.make_pairs(task="ir")
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="chatgpt_ir"))

    pairs = dataset_builder.faqs.make_pairs_mcom()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_mcom"))

    pairs = dataset_builder.faqs.make_pairs_cidades()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_cidades"))

    pairs_dict = dataset_builder.faqs.make_pairs_general()
    for tag, pairs in pairs_dict.items():
        dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name=f"faq_{tag}"))
        assert len(dfs[-1]) >= 2, dfs[-1]

    pairs = dataset_builder.faqs.make_pairs_defesa()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_defesa"))

    pairs = dataset_builder.faqs.make_pairs_mds()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_mds"))

    pairs = dataset_builder.faqs.make_pairs_mdh()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_mdh"))

    pairs = dataset_builder.faqs.make_pairs_mec()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_mec"))

    pairs = dataset_builder.faqs.make_pairs_midr()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_midr"))

    pairs = dataset_builder.faqs.make_pairs_receita_federal()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_receita_federal"))

    pairs = dataset_builder.faqs.make_pairs_portal_da_transp()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_portal_da_transp"))

    pairs = dataset_builder.faqs.make_pairs_senado()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_senado"))

    pairs = dataset_builder.acronyms_and_aliases.make_pairs(long_segments=args.long_segments)
    dfs.append(
        dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="acronyms_and_aliases")
    )

    pairs_a, pairs_b = dataset_builder.leg_bills.make_pairs_fed_bills(debug=args.debug)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs_a, source_name="leg_bills_art_fed"))
    dfs.append(dataset_builder.utils.gen_dataframe(pairs_b, source_name="leg_bills_just_fed"))

    pairs_dict = dataset_builder.leg_bills.make_pairs_state_bills(debug=args.debug)
    for tag, (pairs_a, pairs_b) in pairs_dict.items():
        dfs.append(
            dataset_builder.utils.gen_dataframe(
                pairs_a, source_name=f"leg_bills_art_{tag}", allow_empty=args.debug
            )
        )
        dfs.append(
            dataset_builder.utils.gen_dataframe(
                pairs_b, source_name=f"leg_bills_just_{tag}", allow_empty=True
            )
        )

    pairs = dataset_builder.int_agreements.make_pairs(long_segments=args.long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="int_agreements"))

    save_cache(dfs, output_dir=args.cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate similar pairs of Brazilian legal data from Tesemõ corpus."
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--long-segments", action="store_true")
    parser.add_argument("--cache-dir", default="./processed_data", type=str)
    args = parser.parse_args()
    main(args)
