#!/bin/bash
# Report pages of a PDF with a long blank vertical stretch (run in a Slurm job): measure_blank_pages.sh <pdf>
source /etc/profile.d/modules.sh; module load ghostscript/10.03.1 >/dev/null 2>&1
D=$(mktemp -d); gs -q -dNOPAUSE -dBATCH -sDEVICE=pnggray -r30 -sOutputFile=$D/p%03d.png "$1"
/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus/bin/python - "$D" <<'PY'
import glob, sys, numpy as np
from PIL import Image
fs = sorted(glob.glob(sys.argv[1] + '/p*.png')); print('pages', len(fs))
for f in fs:
    a = np.array(Image.open(f)) < 200; h = a.shape[0]
    rows = a[int(h*0.07):int(h*0.92)].any(axis=1)
    run = best = 0
    for r in rows: run = 0 if r else run + 1; best = max(best, run)
    if best / len(rows) > 0.15: print(f'page {int(f[-7:-4]):3d}: blank {best/len(rows):.0%}')
PY
rm -rf "$D"
