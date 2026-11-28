#!/usr/bin/env bash
# Diagnostic: the Kaggle search grid on seed 2 of G=5 + halo 4 gave genPPL ~191 (entropy 3.25, 31 doubled words
# per 1k words), while Paper 1 scored the same model at 19.6 on Kaggle (sample_eval.py --fast, fp16 autocast).
# This runs Paper 1's own evaluator on that model twice, fp16 (Paper 1's setting) on GPU 0 and fp32 (the search
# grid's setting) on GPU 1, 128 stories each, and prints the library versions.
#   fp16 ~19.6 and fp32 ~190  -> fp32 generation on the T4 is the problem
#   both bad                  -> the Kaggle environment changed since Paper 1's run
# Internet ON. Input: the block-ncm notebook output holding runs/oursc_M4_G5_halo4_s2/model.pt.
#
# Notebook cell:
#   !git clone -q --recursive https://github.com/Rishab-Gupta24/Block-ncm-testing-p2 /kaggle/working/p2repo
#   !bash /kaggle/working/p2repo/kaggle/diag_precision.sh
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
[ -e ~/ncm ] || ln -s $NCM_ROOT ~/ncm
pip install -q omegaconf hydra-core "transformers>=4.51" huggingface_hub regex 2>&1 | tail -2
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)" | tee -a $OUT/progress.txt
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hazyresearch/ncm-checkpoints", allow_patterns=["tinystories/*"], local_dir="$NCM_ROOT/checkpoints")
EOF

PT=$(find /kaggle/input -path "*oursc_M4_G5_halo4_s2/model.pt" | head -1)
[ -n "$PT" ] || { log "NO MODEL FOUND (attach the block-ncm seed-2 notebook output)"; exit 1; }
D=$(dirname $PT)
log "model $D"
cd $REPO/external/block-ncm
gpu=0
for amp in fp16 fp32; do
  (CUDA_VISIBLE_DEVICES=$gpu python -W ignore experiments/sample_eval.py --run $D --M 4 --G 5 --fast \
     --ckpt $NCM_ROOT/checkpoints/tinystories --n 128 --seeds 42 --bs 32 --amp $amp \
     --out $OUT/diag_$amp.json > $OUT/diag_$amp.txt 2>&1
   { echo "=== $amp"; grep -E '^  "(genppl|entropy|boundary_ratio)"' $OUT/diag_$amp.txt || tail -n 30 $OUT/diag_$amp.txt; } | tee -a $OUT/progress.txt) &
  gpu=$((gpu + 1))
done
wait
rm -rf $NCM_ROOT/checkpoints $NCM_ROOT/.git 2>/dev/null
log "DONE"
