"""Catalog integrations for bioacoustic data sources.

Catalogs provide access to external bioacoustic databases and services:
- iNaturalist: Citizen science observations with audio recordings
- Xeno-canto: Bird sound archive
- Macaulay Library: Cornell's multimedia archive
- eBird: Bird observation data
- Species: Taxonomic name lookup
- HuggingFace: Model and dataset hosting
"""

from typing import TYPE_CHECKING

import click

# Lazy imports for CLI performance - services only loaded when commands execute
if TYPE_CHECKING:
    pass


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
@click.option("--has-sounds", is_flag=True, help="Only show observations with sounds")
@click.option("--limit", type=int, default=20, help="Maximum number of results")
@click.option("--output", "-o", default=None, help="Output file path for CSV (optional)")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def inat_search(
    species: str,
    taxon_id: int,
    place_id: int,
    project_id: str,
    quality_grade: str,
    has_sounds: bool,
    limit: int,
    output: str,
    quiet: bool,
) -> None:
    """Search for iNaturalist observations."""
    from bioamla.cli.service_helpers import handle_result, services

    if not species and not taxon_id and not place_id and not project_id:
        raise click.UsageError(
            "At least one search filter must be provided (--species, --taxon-id, --place-id, or --project-id)"
        )

    result = services.inaturalist.search(
        taxon_id=taxon_id,
        taxon_name=species,
        place_id=place_id,
        quality_grade=quality_grade,
        per_page=limit,
    )
    observations = handle_result(result).observations
    if not observations:
        click.echo("No observations found matching the search criteria.")
        return

    if output:
        rows = []
        for obs in observations:
            taxon = obs.get("taxon", {})
            observed_on_raw = obs.get("observed_on", "")
            if hasattr(observed_on_raw, "strftime"):
                observed_on = observed_on_raw.strftime("%Y-%m-%d")
            else:
                observed_on = str(observed_on_raw) if observed_on_raw else ""
            rows.append({
                "observation_id": obs.get("id"),
                "scientific_name": taxon.get("name", ""),
                "common_name": taxon.get("preferred_common_name", ""),
                "sound_count": len(obs.get("sounds", [])),
                "observed_on": observed_on,
                "location": obs.get("place_guess", ""),
                "url": f"https://www.inaturalist.org/observations/{obs.get('id')}",
            })
        fieldnames = [
            "observation_id",
            "scientific_name",
            "common_name",
            "sound_count",
            "observed_on",
            "location",
            "url",
        ]
        services.file.write_csv_dicts(output, rows, fieldnames=fieldnames)
        click.echo(f"Saved {len(observations)} observations to {output}")
    else:
        click.echo(f"\nFound {len(observations)} observations with sounds:\n")
        click.echo(f"{'ID':<12} {'Species':<30} {'Sounds':<8} {'Date':<12} {'Location':<30}")
        click.echo("-" * 95)
        for obs in observations:
            taxon = obs.get("taxon", {})
            obs_id = obs.get("id", "")
            name = taxon.get("name", "Unknown")[:28]
            sound_count = len(obs.get("sounds", []))
            observed_on_raw = obs.get("observed_on", "")
            if hasattr(observed_on_raw, "strftime"):
                observed_on = observed_on_raw.strftime("%Y-%m-%d")
            else:
                observed_on = str(observed_on_raw)[:10] if observed_on_raw else ""
            location = (obs.get("place_guess", "") or "")[:28]
            click.echo(f"{obs_id:<12} {name:<30} {sound_count:<8} {observed_on:<12} {location:<30}")


@catalogs_inat.command("stats")
@click.argument("project_id")
@click.option("--output", "-o", default=None, help="Output file path for JSON (optional)")
@click.option("--quiet", is_flag=True, help="Suppress progress output, print only JSON")
def inat_stats(project_id: str, output: str, quiet: bool) -> None:
    """Get statistics for an iNaturalist project."""
    import json

    from bioamla.cli.service_helpers import handle_result, services

    result = services.inaturalist.get_project_stats(project_id=project_id)
    stats = handle_result(result)

    if output:
        services.file.write_json(output, stats.to_dict())
        click.echo(f"Saved project stats to {output}")
    elif quiet:
        click.echo(json.dumps(stats.to_dict(), indent=2))
    else:
        click.echo(f"\nProject: {stats.title}")
        click.echo(f"URL: {stats.url}")
        click.echo(f"Type: {stats.project_type}")
        if stats.place:
            click.echo(f"Place: {stats.place}")
        click.echo(f"Created: {stats.created_at}")
        click.echo("\nStatistics:")
        click.echo(f"  Observations: {stats.observation_count}")
        click.echo(f"  Species: {stats.species_count}")
        click.echo(f"  Observers: {stats.observers_count}")


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
    help="Filter by sound license(s) (comma-separated: cc0, cc-by, cc-by-nc, cc-by-sa, cc-by-nd, cc-by-nc-sa, cc-by-nc-nd)",
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
    from bioamla.cli.service_helpers import handle_result, services

    # Parse taxon IDs
    taxon_id_list = None
    if taxon_ids:
        taxon_id_list = [int(t.strip()) for t in taxon_ids.split(",")]

    # Parse licenses
    license_list = None
    if license:
        license_list = [lic.strip() for lic in license.split(",")]

    result = services.inaturalist.download(
        output_dir=output_dir,
        taxon_ids=taxon_id_list,
        taxon_name=taxon_name,
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade if quality_grade != "any" else None,
        sound_license=license_list,
        d1=start_date,
        d2=end_date,
        obs_per_taxon=obs_per_taxon,
    )
    download_result = handle_result(result)
    if not quiet:
        click.echo("\nDownload complete:")
        click.echo(f"  Observations: {download_result.total_observations}")
        click.echo(f"  Sounds downloaded: {download_result.total_sounds}")

        # Show explanation if counts differ
        if download_result.observations_with_multiple_sounds > 0:
            click.echo(
                f"    ({download_result.observations_with_multiple_sounds} "
                f"observation{'s' if download_result.observations_with_multiple_sounds > 1 else ''} "
                f"had multiple sound files)"
            )

        if download_result.skipped_existing > 0:
            click.echo(f"  Skipped (existing): {download_result.skipped_existing}")
        if download_result.failed_downloads > 0:
            click.echo(f"  Failed: {download_result.failed_downloads}")
        click.echo(f"  Output directory: {download_result.output_dir}")
        click.echo(f"  Metadata file: {download_result.metadata_file}")


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
    from bioamla.cli.service_helpers import services

    click.echo(f"Pushing model folder {path} to HuggingFace Hub: {repo_id}...")

    result = services.huggingface.push_model(path, repo_id, private=private, commit_message=commit_message)

    if not result.success:
        click.echo(f"Error: {result.error}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)

    click.echo(f"Successfully pushed model to: {result.data.url}")


@catalogs_hf.command("push-dataset")
@click.argument("path")
@click.argument("repo_id")
@click.option(
    "--private/--public", default=False, help="Make the repository private (default: public)"
)
@click.option("--commit-message", default=None, help="Custom commit message for the push")
def hf_push_dataset(path: str, repo_id: str, private: bool, commit_message: str) -> None:
    """Push a dataset folder to the HuggingFace Hub."""
    from bioamla.cli.service_helpers import services

    click.echo(f"Pushing dataset folder {path} to HuggingFace Hub: {repo_id}...")

    result = services.huggingface.push_dataset(path, repo_id, private=private, commit_message=commit_message)

    if not result.success:
        click.echo(f"Error: {result.error}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)

    click.echo(f"Successfully pushed dataset to: {result.data.url}")


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
def xc_search(species: str, genus: str, country: str, quality: str, sound_type: str, max_results: int, output_format: str) -> None:
    """Search Xeno-canto for bird recordings."""
    import json as json_lib

    from bioamla.cli.service_helpers import handle_result, services

    result = services.xeno_canto.search(
        species=species,
        genus=genus,
        country=country,
        quality=quality,
        sound_type=sound_type,
        max_results=max_results,
    )
    recordings = handle_result(result).recordings
    if not recordings:
        click.echo("No recordings found.")
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
        click.echo(f"Found {len(recordings)} recordings:\n")
        for r in recordings:
            click.echo(f"XC{r.id}: {r.scientific_name} ({r.common_name})")
            click.echo(f"  Quality: {r.quality} | Type: {r.sound_type} | Length: {r.length}")
            click.echo(f"  Location: {r.location}, {r.country}")
            click.echo(f"  Recordist: {r.recordist}")
            click.echo(f"  URL: {r.url}")
            click.echo()


@catalogs_xc.command("download")
@click.option("--species", "-s", help="Species name (scientific or common)")
@click.option("--genus", "-g", help="Genus name")
@click.option("--country", "-c", help="Country name")
@click.option("--quality", "-q", default="A", help="Recording quality filter (default: A)")
@click.option("--max-recordings", "-n", default=10, type=int, help="Maximum recordings to download")
@click.option("--output-dir", "-o", default="./xc_recordings", help="Output directory")
@click.option("--delay", default=1.0, type=float, help="Delay between downloads in seconds")
def xc_download(species: str, genus: str, country: str, quality: str, max_recordings: int, output_dir: str, delay: float) -> None:
    """Download recordings from Xeno-canto."""
    from bioamla.cli.service_helpers import handle_result, services

    click.echo("Searching Xeno-canto...")

    result = services.xeno_canto.download(
        species=species,
        genus=genus,
        country=country,
        quality=quality,
        max_recordings=max_recordings,
        output_dir=output_dir,
        delay=delay,
    )
    download_result = handle_result(result)
    if download_result.total == 0:
        click.echo("No recordings found.")
        return

    click.echo(f"\nDownload complete: {download_result.downloaded}/{download_result.total} recordings")


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

    from bioamla.cli.service_helpers import handle_result, services

    result = services.macaulay.search(
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
    recordings = handle_result(result).recordings
    if not recordings:
        click.echo("No recordings found.")
        return

    if output_format == "json":
        click.echo(json_lib.dumps([a.to_dict() for a in recordings], indent=2))
    else:
        click.echo(f"Found {len(recordings)} recordings:\n")
        for a in recordings:
            click.echo(f"ML{a.catalog_id}: {a.scientific_name} ({a.common_name})")
            click.echo(f"  Rating: {a.rating}/5 | Duration: {a.duration or 'N/A'}s")
            click.echo(f"  Location: {a.location}, {a.country}")
            click.echo(f"  Contributor: {a.user_display_name}")
            click.echo()


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
    from bioamla.cli.service_helpers import handle_result, services

    click.echo("Searching Macaulay Library...")

    result = services.macaulay.download(
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
    download_result = handle_result(result)
    if download_result.total == 0:
        click.echo("No recordings found.")
        return

    click.echo(f"\nDownload complete: {download_result.downloaded}/{download_result.total} recordings")


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
    from bioamla.cli.service_helpers import services

    result = services.species.lookup(name, ebird_only=True)

    if not result.success:
        click.echo(f"Species not found in eBird taxonomy: {name}")
        raise SystemExit(1)

    info = result.data
    click.echo(f"Scientific name: {info.scientific_name}")
    click.echo(f"Common name: {info.common_name}")
    click.echo(f"Species code: {info.species_code}")
    click.echo(f"Family: {info.family}")
    click.echo(f"Order: {info.order}")


@catalogs_ebird.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Maximum results")
def ebird_search(query: str, limit: int) -> None:
    """Fuzzy search eBird taxonomy for species."""
    from bioamla.cli.service_helpers import services

    result = services.species.search(query, limit=limit)

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    matches = result.data
    if not matches:
        click.echo(f"No species found matching: {query}")
        return

    click.echo(f"Found {len(matches)} matching species:\n")
    for r in matches:
        score = r.score * 100
        click.echo(f"{r.scientific_name} - {r.common_name} ({r.species_code})")
        click.echo(f"  Family: {r.family} | Match: {score:.0f}%")


@catalogs_ebird.command("validate")
@click.argument("species_code")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option("--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key (or set EBIRD_API_KEY)")
@click.option("--distance", type=float, default=50, help="Search radius in km")
def ebird_validate(species_code: str, lat: float, lng: float, api_key: str, distance: float) -> None:
    """Validate if a species is expected at a location."""
    from bioamla.services.ebird import EBirdService

    service = EBirdService(api_key=api_key)
    result = service.validate_species(
        species_code=species_code,
        lat=lat,
        lng=lng,
        distance_km=distance,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    validation = result.data
    if validation.is_valid:
        click.echo(f"✓ {species_code} is expected at this location")
        click.echo(f"  Found {validation.nearby_observations} nearby observations")
        if validation.most_recent_observation:
            click.echo(f"  Most recent: {validation.most_recent_observation}")
    else:
        click.echo(f"✗ {species_code} not recently observed at this location")
        click.echo(f"  {validation.total_species_in_area} other species observed nearby")


@catalogs_ebird.command("nearby")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option("--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key (or set EBIRD_API_KEY)")
@click.option("--distance", type=float, default=25, help="Search radius in km")
@click.option("--days", type=int, default=14, help="Days back to search")
@click.option("--limit", type=int, default=20, help="Maximum results")
@click.option("--output", "-o", help="Output CSV file")
def ebird_nearby(
    lat: float, lng: float, api_key: str, distance: float, days: int, limit: int, output: str
) -> None:
    """Get recent eBird observations near a location."""
    from pathlib import Path

    from bioamla.cli.service_helpers import services
    from bioamla.services.ebird import EBirdService

    service = EBirdService(api_key=api_key)
    result = service.get_nearby(
        lat=lat,
        lng=lng,
        distance_km=distance,
        days=days,
        limit=limit,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    observations = result.data.observations

    click.echo(f"Found {len(observations)} recent observations:")
    for obs in observations[:10]:
        count_str = f" (x{obs.how_many})" if obs.how_many else ""
        click.echo(f"  {obs.common_name}{count_str} - {obs.location_name}")

    if len(observations) > 10:
        click.echo(f"  ... and {len(observations) - 10} more")

    if output:
        services.file.ensure_directory(Path(output).parent)
        fieldnames = [
            "species_code",
            "common_name",
            "scientific_name",
            "location_name",
            "observation_date",
            "how_many",
        ]
        rows = [obs.to_dict() for obs in observations]
        services.file.write_csv_dicts(output, rows, fieldnames=fieldnames)
        click.echo(f"Saved to: {output}")


