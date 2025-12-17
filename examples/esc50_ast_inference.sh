#example workflow for esc50
bioamla dataset download "https://www.bioamla.org/datasets/scp_small.zip" .
bioamla dataset unzip "scp_small.zip" .
bioamla models predict ast "scp_small" --batch --model-path "esc50/best_model"
