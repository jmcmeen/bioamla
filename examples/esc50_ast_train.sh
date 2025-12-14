#example workflow for esc50
bioamla version
bioamla devices
bioamla ast train \
  --training-dir "esc50" \
  --train-dataset "ashraq/esc50" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 8 \
  --fp16 \
  --gradient-accumulation-steps 2 \
  --dataloader-num-workers 4 \
  --mlflow-experiment-name "esc50"
