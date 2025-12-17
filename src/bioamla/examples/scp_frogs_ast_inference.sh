#example workflow for SCP frogs
bioamla dataset download "https://www.bioamla.org/datasets/scp_small.zip" .
bioamla dataset unzip "scp_small.zip" .
bioamla models ast-predict "scp_small" --batch --model-path "bioamla/scp-frogs"
