#!/usr/bin/env bash
# Phase 2 search grid on the Kaggle-trained seeds of Paper 1's G=5 + halo 4 model (two T4s, one model per GPU).
# Internet ON. Inputs: the block-ncm notebook's outputs that hold runs/oursc_M4_G5_halo4_s*/model.pt
# (Version 7 = seed 1, Version 9 = seed 2). Every attached model.pt matching that pattern is evaluated.
#
# Notebook cell:
#   !git clone -q --recursive https://github.com/Rishab-Gupta24/Block-ncm-testing-p2 /kaggle/working/p2repo
#   !bash /kaggle/working/p2repo/kaggle/run_search.sh
#
# Seconds per story are NOT used from this run (T4, fp32, shared machine): batch-1 timing comes from the local
# RTX 5050 run only. Quality metrics (genPPL, consistency, mistakes) are what this run adds.
set -u
W=/kaggle/working
REPO=$W/p2repo
OUT=$W/results
export NCM_ROOT=$W/ncm PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT
log() { echo "=== $* $(date -u +%H:%M:%S)" | tee -a $OUT/progress.txt; }

log "setup"
nvidia-smi -L | tee $OUT/gpu.txt
[ -d $NCM_ROOT ] || git clone -q https://github.com/HazyResearch/ncm $NCM_ROOT
(cd $NCM_ROOT && git checkout -q c702742)
sed -i '651s/z_hat.add_(z_q_level)/z_hat = z_hat + z_q_level/' $NCM_ROOT/models/quant.py
[ -e ~/ncm ] || ln -s $NCM_ROOT ~/ncm   # block-ncm's error_types.py hard-codes ~/ncm
pip install -q omegaconf hydra-core "transformers>=4.51" huggingface_hub regex 2>&1 | tail -2
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hazyresearch/ncm-checkpoints", revision="d342ce8aaea58830eef3b8ad473664673e41f692", allow_patterns=["tinystories/*"], local_dir="$NCM_ROOT/checkpoints")
EOF

mapfile -t MODELS < <(find /kaggle/input -path "*oursc_M4_G5_halo4_s*/model.pt" | sort)
log "found ${#MODELS[@]} models: ${MODELS[*]}"
[ ${#MODELS[@]} -gt 0 ] || { log "NO MODELS FOUND (attach the block-ncm notebook outputs)"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l)

run_one() {  # run_one GPU MODEL_PT
  local gpu=$1 pt=$2 d name o
  d=$(dirname $pt); name=$(basename $d)_$(echo $pt | md5sum | cut -c1-6); o=$OUT/search_$name
  log "[gpu $gpu] search grid $d"
  (cd $REPO && CUDA_VISIBLE_DEVICES=$gpu python -W ignore experiments/search_grid.py --run $d --timing-reps 1 --rows 32 \
     --out-dir $o > $o.txt 2>&1)
  log "[gpu $gpu] evaluate $d"
  (cd $REPO && CUDA_VISIBLE_DEVICES=$gpu python -W ignore -m p2.evaluate --M 4 --real none \
     --samples $(ls $o/*_samples.json | sed "s|.*/\(.*\)_samples.json|\1=&|") --out $o/eval.json > $o.eval.txt 2>&1)
  if [ -s $o/eval.json ]; then log "[gpu $gpu] ok: $o/eval.json"; else log "[gpu $gpu] FAILED: $o"; for f in $o.txt $o.eval.txt; do tail -n 30 $f; done | tee -a $OUT/progress.txt; fi
}

i=0
for pt in "${MODELS[@]}"; do
  gpu=$((i % NGPU)); run_one $gpu $pt & i=$((i + 1))
  if [ $((i % NGPU)) -eq 0 ]; then wait; fi
done
wait
rm -rf $NCM_ROOT/checkpoints $NCM_ROOT/.git 2>/dev/null   # keep the saved output small
log "DONE"
