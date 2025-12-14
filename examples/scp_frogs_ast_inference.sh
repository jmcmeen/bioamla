#example workflow for SCP frogs
bioamla download "https://www.bioamla.org/datasets/scp_small.zip" .
bioamla unzip "scp_small.zip" .
bioamla ast predict "scp_small" --batch --model-path "scp-frogs/best_model"
