from __future__ import absolute_import, division, print_function

import json
import os

import nlp
from pyarrow import csv

_DESCRIPTION = """ARQMATH dataset"""

_HOMEPAGE = "https://www.cs.rit.edu/~dprl/ARQMath/"

_CITATION = """\
"""

DATA_URL = "https://httpd.test.gipp.com/qa-pair.csv"


class ARQMath(nlp.GeneratorBasedBuilder):
    """ARQMath dataset."""

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features({
                # qID,aID,q,a,rel
                "qID": nlp.Value("string"),
                "aID": nlp.Value("string"),
                "q": nlp.Value("string"),
                "a": nlp.Value("string"),
                "rel": nlp.Value("string"),
            }),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        csv_path = dl_manager.download_and_extract(DATA_URL)

        return [
            nlp.SplitGenerator(name='qa', gen_kwargs={"filepath": csv_path}),
        ]

    def _generate_examples(self, filepath):
        df = csv.read_csv(filepath).to_pandas()

        for idx, row in df.iterrows():
            yield idx, {
                'qID': row['qID'],
                'aID': row['aID'],
                'q': row['q'],
                'a': row['a'],
                'rel': row['rel'],
            }


