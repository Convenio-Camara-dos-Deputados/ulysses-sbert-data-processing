import os
import argparse

import pandas as pd
import colorama

import dataset_builder


def check_data_availability_() -> None:
    for subfolder in ["outros", "legislativo"]:
        cur_path = os.path.join(dataset_builder.utils.Config.TESEMO_PATH, subfolder)
        if not os.path.isdir(cur_path):
            raise FileNotFoundError(
                f"Could not find Tesemõ's submodule '{cur_path}' "
                f"(as per --tesemo-path={dataset_builder.utils.Config.TESEMO_PATH}). "
                f"Please download Tesemõ's submodule '{subfolder}' data "
                f"from {colorama.Fore.YELLOW}"
                "'https://cloud.andrelab.icmc.usp.br/s/b7fP9D5y4z4WsK4'"
                f"{colorama.Style.RESET_ALL}."
            )

    if not os.path.exists(dataset_builder.utils.Config.COMPLEMENTARY_DATADIR):
        raise FileNotFoundError(
            f"Could not find '{dataset_builder.utils.Config.COMPLEMENTARY_DATADIR}' (as per "
            f"--complementary-data-path={dataset_builder.utils.Config.COMPLEMENTARY_DATADIR}). "
            "Please download 'ulysses_sbert_complementary_pair_data.zip' data "
            f"from {colorama.Fore.YELLOW}"
            "'https://cloud.andrelab.icmc.usp.br/s/McLJ6KyWAdKPiEd'"
            f"{colorama.Style.RESET_ALL}."
        )


def build_pairs(tesemo_path: str, complementary_data_path: str, cache_dir: str, long_segments: bool, debug: bool) -> None:
    cache_dir = dataset_builder.utils.expand_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    dataset_builder.utils.Config.TESEMO_PATH = dataset_builder.utils.expand_path(tesemo_path)
    dataset_builder.utils.Config.COMPLEMENTARY_DATADIR = dataset_builder.utils.expand_path(complementary_data_path)

    check_data_availability_()

    dfs: list[pd.DataFrame] = []

    fns = (
        dataset_builder.general.make_pairs_tjac,
        dataset_builder.general.make_pairs_tjal,
        dataset_builder.general.make_pairs_tjam,
        dataset_builder.general.make_pairs_tjap,
        dataset_builder.general.make_pairs_tjba,
        dataset_builder.general.make_pairs_tjce,
        dataset_builder.general.make_pairs_tjdf,
        dataset_builder.general.make_pairs_tjes,
        dataset_builder.general.make_pairs_tjgo,
        dataset_builder.general.make_pairs_tjma,
        dataset_builder.general.make_pairs_tjmg,
        dataset_builder.general.make_pairs_tjms,
        dataset_builder.general.make_pairs_tjmt,
        dataset_builder.general.make_pairs_tjpa,
        dataset_builder.general.make_pairs_tjpb,
        dataset_builder.general.make_pairs_tjpe,
        dataset_builder.general.make_pairs_tjpi,
        dataset_builder.general.make_pairs_tjpr,
        dataset_builder.general.make_pairs_tjrj,
        dataset_builder.general.make_pairs_tjrn,
        dataset_builder.general.make_pairs_tjro,
        dataset_builder.general.make_pairs_tjrr,
        dataset_builder.general.make_pairs_tjrs,
        dataset_builder.general.make_pairs_tjsc,
        dataset_builder.general.make_pairs_tjse,
        dataset_builder.general.make_pairs_tjsp,
        dataset_builder.general.make_pairs_tjto,
        dataset_builder.general.make_pairs_state_ac,
        dataset_builder.general.make_pairs_state_al,
        dataset_builder.general.make_pairs_state_am,
        dataset_builder.general.make_pairs_state_ap,
        dataset_builder.general.make_pairs_state_ba,
        dataset_builder.general.make_pairs_state_ce,
        dataset_builder.general.make_pairs_state_df,
        dataset_builder.general.make_pairs_state_es,
        dataset_builder.general.make_pairs_state_go,
        dataset_builder.general.make_pairs_state_ma,
        dataset_builder.general.make_pairs_state_mg,
        dataset_builder.general.make_pairs_state_ms,
        dataset_builder.general.make_pairs_state_mt,
        dataset_builder.general.make_pairs_state_pa,
        dataset_builder.general.make_pairs_state_pb,
        dataset_builder.general.make_pairs_state_pe,
        dataset_builder.general.make_pairs_state_pi,
        dataset_builder.general.make_pairs_state_pr,
        dataset_builder.general.make_pairs_state_rj,
        dataset_builder.general.make_pairs_state_rn,
        dataset_builder.general.make_pairs_state_ro,
        dataset_builder.general.make_pairs_state_rr,
        dataset_builder.general.make_pairs_state_rs,
        dataset_builder.general.make_pairs_state_sc,
        dataset_builder.general.make_pairs_state_se,
        dataset_builder.general.make_pairs_state_sp,
        dataset_builder.general.make_pairs_state_to,
    )

    for fn in fns:
        pairs, source_name = fn(long_segments=long_segments)
        dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name=source_name))

    dataset_builder.utils.save_cache(dfs, output_dir=cache_dir)

    pairs_a, pairs_b = dataset_builder.leg_bills.make_pairs_fed_bills(long_segments=long_segments, debug=debug)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs_a, source_name="leg_bills_art_fed"))
    dfs.append(dataset_builder.utils.gen_dataframe(pairs_b, source_name="leg_bills_just_fed"))

    dataset_builder.utils.save_cache(dfs, output_dir=cache_dir)

    pairs_dict = dataset_builder.leg_bills.make_pairs_state_bills(
        long_segments=long_segments,
        debug=debug,
    )

    for tag, (pairs_a, pairs_b) in pairs_dict.items():
        dfs.append(dataset_builder.utils.gen_dataframe(pairs_a, source_name=f"leg_bills_art_{tag}", allow_empty=debug))
        dfs.append(dataset_builder.utils.gen_dataframe(pairs_b, source_name=f"leg_bills_just_{tag}", allow_empty=True))

    dataset_builder.utils.save_cache(dfs, output_dir=cache_dir)

    pairs = dataset_builder.acronyms_and_aliases.make_pairs(long_segments=long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="acronyms_and_aliases"))

    fns = (
        dataset_builder.general.make_pairs_tv_camara,
        dataset_builder.general.make_pairs_mpt,
        dataset_builder.general.make_pairs_mpm,
        dataset_builder.general.make_pairs_tcu,
        dataset_builder.general.make_pairs_radio_camara,
        dataset_builder.general.make_pairs_trf4,
        dataset_builder.general.make_pairs_trf1,
        dataset_builder.general.make_pairs_stj,
        dataset_builder.general.make_pairs_bc,
        dataset_builder.general.make_pairs_camara,
        dataset_builder.general.make_pairs_senado,
        dataset_builder.general.make_pairs_stf,
        dataset_builder.general.make_pairs_stm,
        dataset_builder.general.make_pairs_tst,
        dataset_builder.general.make_pairs_tst_radio,
        dataset_builder.general.make_pairs_tst_tv,
        dataset_builder.general.make_pairs_trf6,
        dataset_builder.general.make_pairs_trf5,
        dataset_builder.general.make_pairs_onu,
        dataset_builder.general.make_pairs_capes,
        dataset_builder.general.make_pairs_camara_comments,
        dataset_builder.general.make_pairs_cnmp,
        dataset_builder.general.make_pairs_tse,
        dataset_builder.general.make_pairs_trf2,
        dataset_builder.general.make_pairs_trf3,
        dataset_builder.general.make_pairs_radio_e_tv_justica,
    )

    for fn in fns:
        pairs, source_name = fn(long_segments=long_segments)
        dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name=source_name))

    pairs_dict = dataset_builder.general.make_pairs_ministerios(long_segments=long_segments)
    for tag, pairs in pairs_dict.items():
        dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name=f"news_min_{tag}"))
        assert len(dfs[-1]) >= 2, dfs[-1]

    dataset_builder.utils.save_cache(dfs, output_dir=cache_dir)

    pairs = dataset_builder.stf_annotated_laws.make_pairs_const()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_const"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_lei_9868_1999()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_lei_9868_1999"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_lei_9882_1999()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_lei_9882_1999"))

    pairs = dataset_builder.stf_annotated_laws.make_pairs_oab()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="stf_comments_oab"))

    pairs = dataset_builder.speeches.make_pairs(long_segments=long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="speeches"))

    pairs = dataset_builder.mentions.make_pairs()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="law_mentions"))

    pairs = dataset_builder.chatgpt_data.make_pairs(task="map2doc")
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="chatgpt_map2doc"))

    pairs = dataset_builder.chatgpt_data.make_pairs(task="clusterComments")
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="chatgpt_clusterComments"))

    pairs = dataset_builder.chatgpt_data.make_pairs(task="ir")
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, min_length=1, source_name="chatgpt_ir"))

    pairs = dataset_builder.faqs.make_pairs_bc()
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="faq_bc"))

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

    pairs = dataset_builder.int_agreements.make_pairs(long_segments=long_segments)
    dfs.append(dataset_builder.utils.gen_dataframe(pairs, source_name="int_agreements"))

    dataset_builder.utils.save_cache(dfs, output_dir=cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate similar pairs of Brazilian legal data from Tesemõ corpus.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--long-segments", action="store_true")
    parser.add_argument("--cache-dir", default="./processed_data", type=str)
    parser.add_argument("--tesemo-path", default="./tesemo_v2.1", type=str)
    parser.add_argument("--complementary-data-path", default="./ulysses_sbert_complementary_data", type=str)
    args = parser.parse_args()
    build_pairs(**vars(args))
