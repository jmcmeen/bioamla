#example workflow for esc50
bioamla download "https://www.bioamla.org/datasets/scp_small.zip" "scp_small.zip" 
bioamla unzip "scp_small.zip" . 
bioamla ast-batch-inference "scp_small" --model-path "esc50/best_model"
