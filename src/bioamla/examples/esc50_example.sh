#example workflow for esc50
bioamla version
bioamla devices
bioamla ast-finetune --training-dir "esc50" --train-dataset "ashraq/esc50" --num-train-epochs 1 --per-device-train-batch-size 8
bioamla download "https://www.bioamla.org/datasets/scp_small.zip" "scp_small.zip" 
bioamla unzip "scp_small.zip" . 
bioamla ast-batch-inference "scp_small" --model-path "esc50/best_model"
