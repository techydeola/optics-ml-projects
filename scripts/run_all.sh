#!/usr/bin/env bash
# Runs every training run used in the README. Finished runs are skipped, so it is safe to stop and restart.
# usage: scripts/run_all.sh
set -u
cd "$(dirname "$0")/.." || exit 1

run() {  # run <tag> <bands> <seed> <protocol> <epochs>
  tag=$1; bands=$2; seed=$3; proto=$4; epochs=$5
  out="eurosat/results/keras/${tag}_${bands}_seed${seed}_summary.json"
  if [ -f "$out" ]; then echo "skip ${tag} ${bands} seed ${seed}"; return; fi
  echo "=== ${tag} ${bands} seed ${seed} (protocol ${proto}, up to ${epochs} epochs) ==="
  python eurosat/src/train.py --tag "$tag" --bands "$bands" --seed "$seed" --protocol "$proto" --epochs "$epochs"
}

# protocol B: fixed 40 epochs, cosine learning rate decay (6 configurations x 3 seeds)
for seed in 42 43 44; do
  for bands in all13 rgb no_nir no_swir vis_nir vis_swir; do
    run fixed "$bands" "$seed" B 40
  done
done

# protocol A: early stopping, patience 6
for seed in 42 43 44; do
  for bands in vis_nir vis_swir no_nir no_swir; do
    run band "$bands" "$seed" A 80
  done
done
for seed in 42 43 44; do run main all13 "$seed" A 40; done
run main rgb 42 A 40
run long rgb 43 A 80
run long rgb 44 A 80

echo "ALL DONE"
