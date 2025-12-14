#example workflow for SCP frogs
bioamla version
bioamla devices
bioamla ast-finetune --training-dir "scp-frogs" --num-train-epochs 1
bioamla download "https://www.bioamla.org/datasets/scp_small.zip" "scp_small.zip" 
bioamla unzip "scp_small.zip" . 
bioamla ast-batch-inference "scp_small" --model-path "scp-frogs/best_model"
