import glob
import os
import tqdm

import segmentador

from . import utils


def make_pairs(long_segments: bool = False) -> list[tuple[str, str]]:
    segmenter = segmentador.BERTSegmenter(device="cuda:0")
    pairs: list[tuple[str, str]] = []

    pbar = tqdm.tqdm(
        glob.glob(os.path.joint(utils.TESEMO_PATH, "./outros/o5_acordos_exteriores/*.txt")),
        desc="(int_agreements)",
    )

    for i, uri in enumerate(pbar):
        with open(uri, "r", encoding="utf-8") as f_in:
            text = f_in.read(5000).strip()

        segs = segmenter(text, remove_noise_subsegments=True)

        if len(segs) < 2:
            continue

        if long_segments:
            i = utils.fetch_first_item_index(segs)

            if i < 0:
                continue

            seg_a = "\n".join(segs[:i])
            seg_b = "\n".join(segs[i:])
            if min(len(seg_a), len(seg_b)) >= 200:
                pairs.append((seg_a, seg_b))

        else:
            title = segs[0]
            first_article = utils.remove_spurious_whitespaces(utils.fetch_first_item(segs))

            if first_article and min(len(title), len(first_article)) >= 200 and max(len(title), len(first_article)) <= 800:
                pairs.append((title, first_article))

        if i % 100 == 0 and len(pairs):
            utils.print_example(*pairs[-1])

    return pairs
