#example workflow for SCP frogs
bioamla version
bioamla devices
bioamla models ast-train \
  --training-dir "scp-frogs" \
  --train-dataset "bioamla/scp-frogs-inat-v1" \
  --num-train-epochs 50 \
  --per-device-train-batch-size 8 \
  --fp16 \
  --gradient-accumulation-steps 2 \
  --dataloader-num-workers 4

