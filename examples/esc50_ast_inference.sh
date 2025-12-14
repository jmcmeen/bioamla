#example workflow for esc50
bioamla download "https://www.bioamla.org/datasets/scp_small.zip" .
bioamla unzip "scp_small.zip" .
bioamla ast predict "scp_small" --batch --model-path "esc50/best_model"
