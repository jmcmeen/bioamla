#example workflow for SCP frogs
bioamla version
bioamla devices
bioamla ast-finetune \
  --training-dir "scp-frogs" \
  --train-dataset "bioamla/scp-frogs" \
  --num-train-epochs 50 \
  --per-device-train-batch-size 8 \
  --fp16 \
  --gradient-accumulation-steps 2 \
  --dataloader-num-workers 4 \
  --mlflow-experiment-name "scp-frogs"

