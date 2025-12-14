#!/bin/bash
# iNaturalist Audio Import Examples
# ==================================
# These examples demonstrate how to use the bioamla inat-audio command
# to download audio observations from iNaturalist.

# Example 1: Download bird sounds from the United States
# taxon_id=3 is Aves (birds), place_id=1 is United States
bioamla inat-audio ./bird_sounds \
    --taxon-id 3 \
    --place-id 1 \
    --quality-grade research \
    --max-observations 50

# Example 2: Download frog sounds (Anura) - any license
bioamla inat-audio ./frog_sounds \
    --taxon-name Anura \
    --max-observations 100

# Example 3: Download owl sounds (Strigiformes) from a specific date range
bioamla inat-audio ./owl_sounds \
    --taxon-id 19350 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --max-observations 200

# Example 4: Download sounds from a specific iNaturalist user
bioamla inat-audio ./user_sounds \
    --user-id kueda \
    --max-observations 50

# Example 5: Download sounds without organizing by species (flat directory)
bioamla inat-audio ./mixed_sounds \
    --taxon-name Aves \
    --place-id 1 \
    --no-organize-by-taxon \
    --max-observations 25

# Example 6: Download with custom rate limiting (slower for large downloads)
bioamla inat-audio ./large_dataset \
    --taxon-id 3 \
    --max-observations 500 \
    --delay 2.0

# Example 7: Quiet mode for scripting (minimal output)
bioamla inat-audio ./scripted_download \
    --taxon-name "Strix varia" \
    --max-observations 20 \
    --quiet

# Common taxon IDs:
# -----------------
# 3      = Aves (Birds)
# 20978  = Amphibia (Amphibians)
# 26036  = Anura (Frogs and Toads)
# 19350  = Strigiformes (Owls)
# 7251   = Passeriformes (Songbirds)
# 40151  = Mammalia (Mammals)
# 47208  = Insecta (Insects)
# 47157  = Orthoptera (Crickets, Grasshoppers)

# Common place IDs:
# -----------------
# 1      = United States
# 6712   = Canada
# 7161   = United Kingdom
# 7167   = Australia
# 8057   = Mexico
