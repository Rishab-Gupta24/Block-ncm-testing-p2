#!/usr/bin/env bash
# Score search-grid samples that an earlier Kaggle run already generated (no generation here). Two T4s: plans 1-2 on
# GPU 0, plans 4-8 on GPU 1. Internet ON. Input: the output of the notebook that ran run_search.sh
# (folders results/search_*/ with *_samples.json).
#
# Notebook cell:
#   !git clone -q --recursive https://github.com/Rishab-Gupta24/Block-ncm-testing-p2 /kaggle/working/p2repo
#   !bash /kaggle/working/p2repo/kaggle/run_eval.sh
set -u
W=/kaggle/working
REPO=$W/p2repo
OUT=$W/results
export NCM_ROOT=$W/ncm PYTHONUNBUFFERED=1
mkdir -p $OUT
log() { echo "=== $* $(date -u +%H:%M:%S)" | tee -a $OUT/progress.txt; }

log "setup"
[ -d $NCM_ROOT ] || git clone -q https://github.com/HazyResearch/ncm $NCM_ROOT
(cd $NCM_ROOT && git checkout -q c702742)
sed -i '651s/z_hat.add_(z_q_level)/z_hat = z_hat + z_q_level/' $NCM_ROOT/models/quant.py
[ -e ~/ncm ] || ln -s $NCM_ROOT ~/ncm   # block-ncm's error_types.py hard-codes ~/ncm
pip install -q omegaconf hydra-core "transformers>=4.51" huggingface_hub regex 2>&1 | tail -2
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hazyresearch/ncm-checkpoints", allow_patterns=["tinystories/*"], local_dir="$NCM_ROOT/checkpoints")
EOF

mapfile -t DIRS < <(find /kaggle/input -type d -name "search_*" | sort)
log "found ${#DIRS[@]} sample folders: ${DIRS[*]}"
[ ${#DIRS[@]} -gt 0 ] || { log "NO SAMPLES FOUND (attach the search notebook's output)"; exit 1; }

for d in "${DIRS[@]}"; do
  name=$(basename $d)
  for part in 0 1; do
    if [ $part -eq 0 ]; then pat="n[12]_K"; else pat="n[48]_K"; fi
    o=$OUT/${name}_eval$part
    (cd $REPO && CUDA_VISIBLE_DEVICES=$part python -W ignore -m p2.evaluate --M 4 --real none \
       --samples $(ls $d/*_samples.json | grep "/$pat" | sed "s|.*/\(.*\)_samples.json|\1=&|") \
       --out $o.json > $o.txt 2>&1; \
     if [ -s $o.json ]; then log "ok: $o.json"; else log "FAILED: $o"; tail -n 30 $o.txt | tee -a $OUT/progress.txt; fi) &
  done
  wait
  cp $d/search_summary.json $OUT/${name}_search_summary.json 2>/dev/null
done
rm -rf $NCM_ROOT/checkpoints $NCM_ROOT/.git 2>/dev/null
log "DONE"
