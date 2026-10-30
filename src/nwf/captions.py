import hashlib
import json

import numpy as np
import pandas as pd

from nwf import config
from nwf.paths import CAPTIONS
from nwf.paths import RAW

if __name__ == "__main__":
    source = config.CAPTION_SOURCE
    path = RAW/"captions"/source["file"].split("/")[-1]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source["sha256"],"the caption file differs from the pinned one"
    table = pd.read_parquet(path).sort_values("cocoid").reset_index(drop=True)
    # one caption per image so no picture weighs more than another
    picked = np.random.default_rng(config.CAPTION_SEED).choice(len(table),config.CAPTIONS,replace=False)
    lines = [" ".join(str(table["sentences"][i][0]).split()) for i in picked]
    CAPTIONS.parent.mkdir(parents=True,exist_ok=True)
    CAPTIONS.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    record = dict(source,images=len(table),picked=len(lines),seed=config.CAPTION_SEED,captions_sha256=hashlib.sha256(CAPTIONS.read_bytes()).hexdigest())
    CAPTIONS.with_name("captions_source.json").write_text(json.dumps(record,indent=1)+"\n",encoding="utf-8",newline="\n")
    print(f"wrote {len(lines)} captions from {len(table)} images")
