"""Catalog integrations for bioacoustic data sources.

Catalogs provide access to external bioacoustic databases and services:
- iNaturalist: Citizen science observations with audio recordings
- Xeno-canto: Bird sound archive
- Macaulay Library: Cornell's multimedia archive
- eBird: Bird observation data
- Species: Taxonomic name lookup
- HuggingFace: Model and dataset hosting
"""

import click


@click.group()
def catalogs() -> None:
    """Access bioacoustic data catalogs and external services."""
    pass


# =============================================================================
# iNaturalist subgroup
# =============================================================================


@catalogs.group("inat")
def catalogs_inat() -> None:
    """iNaturalist observation database."""
    pass


@catalogs_inat.command("search")
@click.option("--species", "-s", default=None, help="Species scientific name to filter by")
@click.option("--taxon-id", "-t", default=None, type=int, help="iNaturalist taxon ID")
@click.option("--place-id", "-p", default=None, type=int, help="iNaturalist place ID")
@click.option("--project-id", default=None, help="iNaturalist project slug or ID")
@click.option(
    "--quality-grade", default="research", help="Quality grade: research, needs_id, or casual"
)
@click.option("--limit", type=int, default=20, help="Maximum number of results")
@click.option("--output", "-o", default=None, help="Output file path for CSV (optional)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def inat_search(
    species: str,
    taxon_id: int,
    place_id: int,
    project_id: str,
    quality_grade: str,
    limit: int,
    output: str,
    quiet: bool,
) -> None:
    """Search for iNaturalist observations."""
    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    found = run_operation(
        "catalogs.inat.search",
        species=species,
        taxon_id=taxon_id,
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade,
        limit=limit,
        output=output,
    )
    rows = found.observations
    if emit_result(found.to_dict()):
        return
    if not rows:
        echo("No observations found matching the search criteria.", style=DIM_STYLE)
        return
    if output:
        print_success(f"Saved {len(rows)} observations to {output}")
        return

    print_header(f"\nFound {len(rows)} observations with sounds:\n")
    echo(f"{'ID':<12} {'Species':<30} {'Sounds':<8} {'Date':<12} {'Location':<30}")
    echo("-" * 95)
    for row in rows:
        echo(
            f"{row['observation_id'] or '':<12} {(row['scientific_name'] or 'Unknown')[:28]:<30} "
            f"{row['sound_count']:<8} {row['observed_on'][:10]:<12} "
            f"{(row['location'] or '')[:28]:<30}"
        )


@catalogs_inat.command("stats")
@click.argument("project_id")
@click.option("--output", "-o", default=None, help="Output file path for JSON (optional)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output, print only JSON")
def inat_stats(project_id: str, output: str, quiet: bool) -> None:
    """Get statistics for an iNaturalist project."""
    import json

    from bioamla.cli.console import print_header, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    report = run_operation("catalogs.inat.stats", project_id=project_id, output=output)
    stats = report.stats
    if emit_result(report.to_dict()):
        return
    if output:
        print_success(f"Saved project stats to {output}")
    elif quiet:
        click.echo(json.dumps(stats.to_dict(), indent=2))
    else:
        print_header(f"\nProject: {stats.title}")
        print_kv("URL", stats.url)
        print_kv("Type", stats.project_type)
        if stats.place:
            print_kv("Place", stats.place)
        print_kv("Created", stats.created_at)
        print_header("\nStatistics:")
        print_kv("  Observations", stats.observation_count)
        print_kv("  Species", stats.species_count)
        print_kv("  Observers", stats.observers_count)


@catalogs_inat.command("download")
@click.argument("output_dir")
@click.option("--taxon-ids", "-t", default=None, help="Comma-separated taxon IDs")
@click.option("--taxon-name", "-n", default=None, help="Taxon name to search for")
@click.option("--place-id", "-p", default=None, type=int, help="iNaturalist place ID")
@click.option("--project-id", default=None, help="iNaturalist project slug or ID")
@click.option(
    "--quality-grade",
    "-q",
    type=click.Choice(["research", "needs_id", "casual", "any"]),
    default="research",
    help="Quality grade filter",
)
@click.option(
    "--license",
    "-l",
    default=None,
    help="Filter by sound license(s) (comma-separated: cc0, cc-by, cc-by-nc, cc-by-sa, "
    "cc-by-nd, cc-by-nc-sa, cc-by-nc-nd)",
)
@click.option("--start-date", "-d1", default=None, help="Start date for observations (YYYY-MM-DD)")
@click.option("--end-date", "-d2", default=None, help="End date for observations (YYYY-MM-DD)")
@click.option("--obs-per-taxon", default=10, type=int, help="Max observations per taxon")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def inat_download(
    output_dir: str,
    taxon_ids: str,
    taxon_name: str,
    place_id: int,
    project_id: str,
    quality_grade: str,
    license: str,
    start_date: str,
    end_date: str,
    obs_per_taxon: int,
    quiet: bool,
) -> None:
    """Download audio observations from iNaturalist."""
    from bioamla.cli.console import echo, print_header, print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    try:
        taxon_id_list = [int(t.strip()) for t in taxon_ids.split(",")] if taxon_ids else None
    except ValueError:
        raise click.BadParameter(
            "expected comma-separated numbers", param_hint="--taxon-ids"
        ) from None
    download_result = run_operation(
        "catalogs.inat.download",
        output_dir=output_dir,
        taxon_ids=taxon_id_list,
        taxon_name=taxon_name,
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade,
        licenses=[lic.strip() for lic in license.split(",")] if license else None,
        start_date=start_date,
        end_date=end_date,
        obs_per_taxon=obs_per_taxon,
    )
    if emit_result(download_result):
        return
    if not quiet:
        print_header("\nDownload complete:")
        print_kv("  Observations", download_result.total_observations)
        print_kv("  Sounds downloaded", download_result.total_sounds)

        if download_result.observations_with_multiple_sounds > 0:
            echo(
                f"    ({download_result.observations_with_multiple_sounds} "
                f"observation{'s' if download_result.observations_with_multiple_sounds > 1 else ''} "
                f"had multiple sound files)"
            )

        if download_result.skipped_existing > 0:
            print_kv("  Skipped (existing)", download_result.skipped_existing)
        if download_result.failed_downloads > 0:
            print_kv("  Failed", download_result.failed_downloads)
        print_kv("  Output directory", download_result.output_dir)
        print_kv("  Metadata file", download_result.metadata_file)


# =============================================================================
# HuggingFace Hub subgroup
# =============================================================================


@catalogs.group("hf")
def catalogs_hf() -> None:
    """HuggingFace Hub model and dataset management."""
    pass


@catalogs_hf.command("push-model")
@click.argument("path")
@click.argument("repo_id")
@click.option(
    "--private/--public", default=False, help="Make the repository private (default: public)"
)
@click.option("--commit-message", default=None, help="Custom commit message for the push")
def hf_push_model(path: str, repo_id: str, private: bool, commit_message: str) -> None:
    """Push a model folder to the HuggingFace Hub."""
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    echo(f"Pushing model folder {path} to HuggingFace Hub: {repo_id}...")
    try:
        result = run_operation(
            "catalogs.hf.push-model",
            path=path,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
        )
    except click.ClickException as e:
        _login_hint(e)
        raise
    if emit_result(result):
        return
    print_success(f"Successfully pushed model to: {result.url}")


def _login_hint(error: click.ClickException) -> None:
    """Suggest logging in when the push itself failed (not for a bad option)."""
    from bioamla.cli.console import print_warning
    from bioamla.exceptions import InvalidInputError

    if not isinstance(error, click.UsageError) and not isinstance(
        error.__cause__, InvalidInputError
    ):
        print_warning("Make sure you are logged in with 'huggingface-cli login'.")


@catalogs_hf.command("push-dataset")
@click.argument("path")
@click.argument("repo_id")
@click.option(
    "--private/--public", default=False, help="Make the repository private (default: public)"
)
@click.option("--commit-message", default=None, help="Custom commit message for the push")
@click.option(
    "--card/--no-card",
    default=True,
    help="Write a README.md dataset card from the manifest/metadata before pushing",
)
@click.option(
    "--coordinates",
    type=click.Choice(["keep", "round", "strip"]),
    default="keep",
    show_default=True,
    help="Per-file latitude/longitude in the uploaded metadata: keep, round, or strip",
)
@click.option(
    "--coordinate-decimals",
    type=click.IntRange(0, 6),
    default=2,
    show_default=True,
    help="Decimal places kept by --coordinates round (2 ~ 1 km, 1 ~ 11 km)",
)
def hf_push_dataset(
    path: str,
    repo_id: str,
    private: bool,
    commit_message: str,
    card: bool,
    coordinates: str,
    coordinate_decimals: int,
) -> None:
    """Push a dataset folder to the HuggingFace Hub.

    When ``--card`` is set (default) and the folder has a ``dataset.json`` manifest
    or a ``metadata.csv``, a ``README.md`` dataset card is generated first; folders
    without either are pushed as-is.

    ``--coordinates round`` or ``strip`` redacts recorder locations in what is
    uploaded (metadata CSVs, manifest sites, generated card, and the position in
    WAV recorder metadata); the local folder keeps full precision.
    """
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    echo(f"Pushing dataset folder {path} to HuggingFace Hub: {repo_id}...")
    # The card and any location warnings print as notices before the upload starts.
    try:
        pushed = run_operation(
            "catalogs.hf.push-dataset",
            path=path,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
            card=card,
            coordinates=coordinates,
            coordinate_decimals=coordinate_decimals,
        )
    except click.ClickException as e:
        _login_hint(e)
        raise
    if json_mode():
        emit_result(pushed.to_dict())
        return
    print_success(f"Successfully pushed dataset to: {pushed.result.url}")


@catalogs_hf.command("pull-dataset")
@click.argument("repo_id")
@click.argument("dest")
@click.option("--split", default=None, help="Single split to pull (default: all splits)")
@click.option("--config", default=None, help="Dataset config/subset name (e.g. HSN for BirdSet)")
@click.option("--audio-column", default=None, help="Override the auto-detected audio column")
@click.option("--label-column", default=None, help="Override the auto-detected label column")
@click.option(
    "--sample-rate", type=int, default=16000, help="Resample clips to this rate (0 to keep source)"
)
@click.option(
    "--layout",
    type=click.Choice(["both", "audiofolder", "flat"]),
    default="both",
    help="Output layout: label subdirs + metadata.csv (both), subdirs only, or flat",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def hf_pull_dataset(
    repo_id: str,
    dest: str,
    split: str,
    config: str,
    audio_column: str,
    label_column: str,
    sample_rate: int,
    layout: str,
    quiet: bool,
) -> None:
    """Pull a HuggingFace audio dataset into the local labeled-folder layout.

    Materializes REPO_ID (e.g. ``ashraq/esc-50``) under DEST as label subdirs and
    a ``metadata.csv``, ready for ``dataset partition`` / ``models ast train``.
    """
    from bioamla.cli.console import echo, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    if not quiet:
        echo(f"Pulling dataset {repo_id} from HuggingFace Hub into {dest}...")
    result = run_operation(
        "catalogs.hf.pull-dataset",
        repo_id=repo_id,
        dest=dest,
        split=split,
        config=config,
        audio_column=audio_column,
        label_column=label_column,
        sample_rate=sample_rate,
        layout=layout,
        verbose=not quiet,
    )
    if emit_result(result):
        return
    print_success(
        f"Wrote {result.files_written} clips across {len(result.labels)} labels to {result.dest}"
    )
    if result.metadata_file:
        print_kv("Metadata", result.metadata_file)


def _format_size(size_bytes: float) -> str:
    """Format bytes into a human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@catalogs_hf.command("cache")
@click.option("--models", is_flag=True, help="Target cached models")
@click.option("--datasets", is_flag=True, help="Target cached datasets")
@click.option("--all", "target_all", is_flag=True, help="Target both models and datasets")
@click.option("--purge", "do_purge", is_flag=True, help="Delete the matched cache (default: list)")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt when purging")
def hf_cache(models: bool, datasets: bool, target_all: bool, do_purge: bool, yes: bool) -> None:
    """Inspect or purge the local HuggingFace cache (datasets/models).

    Datasets and models pulled/loaded from the Hub are cached so repeat grabs are
    fast. Lists the cache by default; pass --purge to reclaim the space.

    Examples:
        bioamla catalogs hf cache                       # list everything cached
        bioamla catalogs hf cache --datasets --purge -y # free dataset cache
    """
    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success, print_warning
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, fail, json_mode

    # Default to both types unless one is singled out.
    if target_all or (not models and not datasets):
        models = datasets = True
    if json_mode() and do_purge and not yes:
        raise click.UsageError("--purge with --json needs --yes (there is no prompt).")

    listing = run_operation("catalogs.hf.cache", models=models, datasets=datasets)
    repos = listing["repos"]

    if json_mode():
        report = {**listing, "purged": None}
        if do_purge and repos:
            report["purged"] = run_operation(
                "catalogs.hf.cache-purge", models=models, datasets=datasets
            )
            if report["purged"].failures:
                fail(
                    f"Could not purge {len(report['purged'].failures)} cached repo(s)",
                    "CatalogError",
                    details=report,
                )
        emit_result(report)
        return

    if not repos:
        echo("No cached data found.", style=DIM_STYLE)
        return

    for r in repos:
        echo(f"  {r.repo_type:>7}: {r.repo_id} ({_format_size(r.size_bytes)})")
    print_header(f"Total: {_format_size(listing['total_bytes'])} across {len(repos)} repo(s)")

    if not do_purge:
        echo("(run with --purge to delete)", style=DIM_STYLE)
        return

    if not yes and not click.confirm("Delete this cached data?"):
        echo("Aborted.", style=DIM_STYLE)
        return

    result = run_operation("catalogs.hf.cache-purge", models=models, datasets=datasets)
    print_success(f"Purged {result.deleted} repo(s), freed {_format_size(result.freed_bytes)}.")
    for failure in result.failures:
        print_warning(f"  warning: {failure}")


# =============================================================================
# Xeno-canto subgroup
# =============================================================================


@catalogs.group("xc")
def catalogs_xc() -> None:
    """Xeno-canto bird recording database."""
    pass


@catalogs_xc.command("search")
@click.option("--species", "-s", help="Species name (scientific or common)")
@click.option("--genus", "-g", help="Genus name")
@click.option("--country", "-c", help="Country name")
@click.option("--quality", "-q", help="Recording quality (A, B, C, D, E)")
@click.option("--type", "sound_type", help="Sound type (song, call, etc.)")
@click.option("--max-results", "-n", default=10, type=int, help="Maximum results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
def xc_search(
    species: str,
    genus: str,
    country: str,
    quality: str,
    sound_type: str,
    max_results: int,
    output_format: str,
) -> None:
    """Search Xeno-canto for bird recordings.

    \b
    Examples:
        # Songs of a species in the US, top 20
        bioamla catalogs xc search --species "Turdus migratorius" \\
            --country "United States" --type song --max-results 20
        # By genus, JSON output for scripting
        bioamla catalogs xc search --genus Turdus --quality A --format json
    """
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "catalogs.xc.search",
        species=species,
        genus=genus,
        country=country,
        quality=quality,
        sound_type=sound_type,
        max_results=max_results,
    )
    recordings = result.recordings
    if emit_result({"count": len(recordings), "recordings": [r.to_dict() for r in recordings]}):
        return
    if not recordings:
        echo("No recordings found.", style=DIM_STYLE)
        return

    if output_format == "json":
        click.echo(json_lib.dumps([r.to_dict() for r in recordings], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=recordings[0].to_dict().keys())
        writer.writeheader()
        for r in recordings:
            writer.writerow(r.to_dict())
    else:
        print_header(f"Found {len(recordings)} recordings:\n")
        for r in recordings:
            print_header(f"XC{r.id}: {r.scientific_name} ({r.common_name})")
            echo(f"  Quality: {r.quality} | Type: {r.sound_type} | Length: {r.length}")
            print_kv("  Location", f"{r.location}, {r.country}")
            print_kv("  Recordist", r.recordist)
            print_kv("  URL", r.url)
            echo()


@catalogs_xc.command("download")
@click.option("--species", "-s", help="Species name (scientific or common)")
@click.option("--genus", "-g", help="Genus name")
@click.option("--country", "-c", help="Country name")
@click.option("--quality", "-q", default="A", help="Recording quality filter (default: A)")
@click.option("--max-recordings", "-n", default=10, type=int, help="Maximum recordings to download")
@click.option("--output-dir", "-o", default="./xc_recordings", help="Output directory")
@click.option("--delay", default=1.0, type=float, help="Delay between downloads in seconds")
def xc_download(
    species: str,
    genus: str,
    country: str,
    quality: str,
    max_recordings: int,
    output_dir: str,
    delay: float,
) -> None:
    """Download recordings from Xeno-canto."""
    from bioamla.cli.console import DIM_STYLE, echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    echo("Searching Xeno-canto...")
    download_result = run_operation(
        "catalogs.xc.download",
        output_dir=output_dir,
        species=species,
        genus=genus,
        country=country,
        quality=quality,
        max_recordings=max_recordings,
        delay=delay,
    )
    if emit_result(download_result):
        return
    if download_result.total == 0:
        echo("No recordings found.", style=DIM_STYLE)
        return

    echo()
    print_success(
        f"Download complete: {download_result.downloaded}/{download_result.total} recordings"
    )


# =============================================================================
# Macaulay Library subgroup
# =============================================================================


@catalogs.group("ml")
def catalogs_ml() -> None:
    """Macaulay Library audio recordings database.

    Use 'catalogs ebird species' or 'catalogs ebird search' to look up species codes.
    """
    pass


@catalogs_ml.command("search")
@click.option("--species-code", "-s", help="eBird species code (e.g., amerob)")
@click.option("--scientific-name", help="Scientific name")
@click.option("--common-name", help="Common name")
@click.option("--region", "-r", help="Region code (e.g., US-NY)")
@click.option("--country", help="Country code (e.g., US)")
@click.option("--taxon-code", help="eBird taxon code for broader searches")
@click.option("--hotspot-code", help="eBird hotspot code")
@click.option("--min-rating", default=0, type=int, help="Minimum quality rating (1-5)")
@click.option("--max-results", "-n", default=10, type=int, help="Maximum results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def ml_search(
    species_code: str,
    scientific_name: str,
    common_name: str,
    region: str,
    country: str,
    taxon_code: str,
    hotspot_code: str,
    min_rating: int,
    max_results: int,
    output_format: str,
) -> None:
    """Search Macaulay Library for audio recordings.

    Requires at least one filter: species-code, scientific-name, common-name,
    region, taxon-code, or hotspot-code.

    Use 'catalogs ebird species <name>' to look up species codes.
    """
    import json as json_lib

    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation(
        "catalogs.ml.search",
        species_code=species_code,
        scientific_name=scientific_name,
        common_name=common_name,
        region=region,
        country=country,
        taxon_code=taxon_code,
        hotspot_code=hotspot_code,
        min_rating=min_rating,
        max_results=max_results,
    )
    recordings = result.recordings
    if emit_result({"count": len(recordings), "recordings": [r.to_dict() for r in recordings]}):
        return
    if not recordings:
        echo("No recordings found.", style=DIM_STYLE)
        return

    if output_format == "json":
        click.echo(json_lib.dumps([a.to_dict() for a in recordings], indent=2))
    else:
        print_header(f"Found {len(recordings)} recordings:\n")
        for a in recordings:
            print_header(f"ML{a.catalog_id}: {a.scientific_name} ({a.common_name})")
            echo(f"  Rating: {a.rating}/5 | Duration: {a.duration or 'N/A'}s")
            print_kv("  Location", f"{a.location}, {a.country}")
            print_kv("  Contributor", a.user_display_name)
            echo()


@catalogs_ml.command("download")
@click.option("--species-code", "-s", help="eBird species code (e.g., amerob)")
@click.option("--scientific-name", help="Scientific name")
@click.option("--common-name", help="Common name")
@click.option("--region", "-r", help="Region code (e.g., US-NY)")
@click.option("--country", help="Country code (e.g., US)")
@click.option("--taxon-code", help="eBird taxon code for broader searches")
@click.option("--hotspot-code", help="eBird hotspot code")
@click.option("--min-rating", default=3, type=int, help="Minimum quality rating (default: 3)")
@click.option("--max-recordings", "-n", default=10, type=int, help="Maximum recordings to download")
@click.option("--output-dir", "-o", default="./ml_recordings", help="Output directory")
def ml_download(
    species_code: str,
    scientific_name: str,
    common_name: str,
    region: str,
    country: str,
    taxon_code: str,
    hotspot_code: str,
    min_rating: int,
    max_recordings: int,
    output_dir: str,
) -> None:
    """Download recordings from Macaulay Library.

    Requires at least one filter: species-code, scientific-name, common-name,
    region, taxon-code, or hotspot-code.

    Use 'catalogs ebird species <name>' to look up species codes.
    """
    from bioamla.cli.console import DIM_STYLE, echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    echo("Searching Macaulay Library...")
    download_result = run_operation(
        "catalogs.ml.download",
        species_code=species_code,
        scientific_name=scientific_name,
        common_name=common_name,
        region=region,
        country=country,
        taxon_code=taxon_code,
        hotspot_code=hotspot_code,
        min_rating=min_rating,
        max_recordings=max_recordings,
        output_dir=output_dir,
    )
    if emit_result(download_result):
        return
    if download_result.total == 0:
        echo("No recordings found.", style=DIM_STYLE)
        return

    echo()
    print_success(
        f"Download complete: {download_result.downloaded}/{download_result.total} recordings"
    )


# =============================================================================
# eBird subgroup
# =============================================================================


@catalogs.group("ebird")
def catalogs_ebird() -> None:
    """eBird bird observation data and taxonomy."""
    pass


@catalogs_ebird.command("species")
@click.argument("name")
def ebird_species(name: str) -> None:
    """Look up species in eBird taxonomy.

    NAME can be a common name, scientific name, or species code.
    """
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    info = run_operation("catalogs.ebird.species", name=name)
    if emit_result(info):
        return
    print_kv("Scientific name", info.scientific_name)
    print_kv("Common name", info.common_name)
    print_kv("Species code", info.species_code)
    print_kv("Family", info.family)
    print_kv("Order", info.order)


@catalogs_ebird.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Maximum results")
def ebird_search(query: str, limit: int) -> None:
    """Fuzzy search eBird taxonomy for species."""
    from bioamla.cli.console import DIM_STYLE, echo, print_header
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation("catalogs.ebird.search", query=query, limit=limit)
    matches = result["matches"]
    if emit_result(result):
        return
    if not matches:
        echo(f"No species found matching: {query}", style=DIM_STYLE)
        return

    print_header(f"Found {len(matches)} matching species:\n")
    for r in matches:
        score = r.score * 100
        print_header(f"{r.scientific_name} - {r.common_name} ({r.species_code})")
        echo(f"  Family: {r.family} | Match: {score:.0f}%")


@catalogs_ebird.command("validate")
@click.argument("species_code")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option(
    "--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key (or set EBIRD_API_KEY)"
)
@click.option("--distance", type=float, default=50, help="Search radius in km")
def ebird_validate(
    species_code: str, lat: float, lng: float, api_key: str, distance: float
) -> None:
    """Validate if a species is expected at a location."""
    from bioamla.cli.console import echo, print_kv, print_success, print_warning
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    validation = run_operation(
        "catalogs.ebird.validate",
        secrets={"EBIRD_API_KEY": api_key},
        species_code=species_code,
        lat=lat,
        lng=lng,
        distance=distance,
    )
    if emit_result(validation):
        return
    if validation.is_valid:
        print_success(f"{species_code} is expected at this location")
        echo(f"  Found {validation.nearby_observations} nearby observations")
        if validation.most_recent_observation:
            print_kv("  Most recent", validation.most_recent_observation)
    else:
        print_warning(f"{species_code} not recently observed at this location")
        echo(f"  {validation.total_species_in_area} other species observed nearby")


@catalogs_ebird.command("nearby")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option(
    "--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key (or set EBIRD_API_KEY)"
)
@click.option("--distance", type=float, default=25, help="Search radius in km")
@click.option("--days", type=int, default=14, help="Days back to search")
@click.option("--limit", type=int, default=20, help="Maximum results")
@click.option("--output", "-o", help="Output CSV file")
def ebird_nearby(
    lat: float, lng: float, api_key: str, distance: float, days: int, limit: int, output: str
) -> None:
    """Get recent eBird observations near a location."""
    from bioamla.cli.console import DIM_STYLE, echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    result = run_operation(
        "catalogs.ebird.nearby",
        secrets={"EBIRD_API_KEY": api_key},
        lat=lat,
        lng=lng,
        distance=distance,
        days=days,
        limit=limit,
        output=output,
    )
    observations = result.observations
    if not json_mode():
        print_header(f"Found {len(observations)} recent observations:")
        for obs in observations[:10]:
            count_str = f" (x{obs.how_many})" if obs.how_many else ""
            echo(f"  {obs.common_name}{count_str} - {obs.location_name}")
        if len(observations) > 10:
            echo(f"  ... and {len(observations) - 10} more", style=DIM_STYLE)
    if emit_result(result.to_dict()):
        return
    if output:
        print_success(f"Saved to: {output}")
