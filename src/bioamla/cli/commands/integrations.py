"""External service integrations (iNaturalist, Xeno-canto, HuggingFace, eBird, etc.)."""

import click

from bioamla.core.files import TextFile


@click.group()
def services():
    """External service integrations (iNaturalist, Xeno-canto, HuggingFace, eBird, etc.)."""
    pass


# =============================================================================
# iNaturalist subgroup
# =============================================================================


@services.group("inat")
def services_inat():
    """iNaturalist observation database."""
    pass


@services_inat.command("search")
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
):
    """Search for iNaturalist observations."""
    from bioamla.services.inaturalist import INaturalistService

    if not species and not taxon_id and not place_id and not project_id:
        raise click.UsageError(
            "At least one search filter must be provided (--species, --taxon-id, --place-id, or --project-id)"
        )

    service = INaturalistService()
    result = service.search(
        taxon_id=taxon_id,
        taxon_name=species,
        place_id=place_id,
        quality_grade=quality_grade,
        per_page=limit,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    observations = result.data.observations
    if not observations:
        click.echo("No observations found matching the search criteria.")
        return

    if output:
        import csv

        with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "observation_id",
                "scientific_name",
                "common_name",
                "sound_count",
                "observed_on",
                "location",
                "url",
            ]
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            for obs in observations:
                taxon = obs.get("taxon", {})
                observed_on_raw = obs.get("observed_on", "")
                if hasattr(observed_on_raw, "strftime"):
                    observed_on = observed_on_raw.strftime("%Y-%m-%d")
                else:
                    observed_on = str(observed_on_raw) if observed_on_raw else ""
                writer.writerow(
                    {
                        "observation_id": obs.get("id"),
                        "scientific_name": taxon.get("name", ""),
                        "common_name": taxon.get("preferred_common_name", ""),
                        "sound_count": len(obs.get("sounds", [])),
                        "observed_on": observed_on,
                        "location": obs.get("place_guess", ""),
                        "url": f"https://www.inaturalist.org/observations/{obs.get('id')}",
                    }
                )
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


@services_inat.command("stats")
@click.argument("project_id")
@click.option("--output", "-o", default=None, help="Output file path for JSON (optional)")
@click.option("--quiet", is_flag=True, help="Suppress progress output, print only JSON")
def inat_stats(project_id: str, output: str, quiet: bool):
    """Get statistics for an iNaturalist project."""
    import json

    from bioamla.services.inaturalist import INaturalistService

    service = INaturalistService()
    result = service.get_project_stats(project_id=project_id)

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    stats = result.data

    if output:
        with TextFile(output, mode="w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f.handle, indent=2)
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


@services_inat.command("download")
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
@click.option("--obs-per-taxon", default=10, type=int, help="Max observations per taxon")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
def inat_download(
    output_dir: str,
    taxon_ids: str,
    taxon_name: str,
    place_id: int,
    project_id: str,
    quality_grade: str,
    obs_per_taxon: int,
    quiet: bool,
):
    """Download audio observations from iNaturalist."""
    from bioamla.services.inaturalist import INaturalistService

    # Parse taxon IDs
    taxon_id_list = None
    if taxon_ids:
        taxon_id_list = [int(t.strip()) for t in taxon_ids.split(",")]

    service = INaturalistService()
    result = service.download(
        output_dir=output_dir,
        taxon_ids=taxon_id_list,
        taxon_name=taxon_name,
        place_id=place_id,
        project_id=project_id,
        quality_grade=quality_grade if quality_grade != "any" else None,
        obs_per_taxon=obs_per_taxon,
    )

    if not result.success:
        click.echo(f"Error: {result.error}")
        raise SystemExit(1)

    download_result = result.data
    if not quiet:
        click.echo(f"\nDownload complete:")
        click.echo(f"  Files downloaded: {download_result.files_downloaded}")
        click.echo(f"  Total size: {download_result.total_bytes / 1024 / 1024:.1f} MB")
        click.echo(f"  Output directory: {output_dir}")


# =============================================================================
# HuggingFace Hub subgroup
# =============================================================================


def _get_folder_size(path: str, limit: int | None = None) -> int:
    """Calculate the total size of a folder in bytes."""
    import os

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
                if limit is not None and total_size > limit:
                    return total_size
    return total_size


def _count_files(path: str, limit: int | None = None) -> int:
    """Count the total number of files in a folder."""
    import os

    count = 0
    for _dirpath, _dirnames, filenames in os.walk(path):
        count += len(filenames)
        if limit is not None and count > limit:
            return count
    return count


def _is_large_folder(
    path: str, size_threshold_gb: float = 5.0, file_count_threshold: int = 1000
) -> bool:
    """Determine if a folder should be uploaded using upload_large_folder."""
    size_threshold_bytes = int(size_threshold_gb * 1024 * 1024 * 1024)
    file_count = _count_files(path, limit=file_count_threshold)
    if file_count > file_count_threshold:
        return True
    folder_size = _get_folder_size(path, limit=size_threshold_bytes)
    return folder_size > size_threshold_bytes


@services.group("hf")
def services_hf():
    """HuggingFace Hub model and dataset management."""
    pass


@services_hf.command("push-model")
@click.argument("path")
@click.argument("repo_id")
@click.option(
    "--private/--public", default=False, help="Make the repository private (default: public)"
)
@click.option("--commit-message", default=None, help="Custom commit message for the push")
def hf_push_model(path: str, repo_id: str, private: bool, commit_message: str):
    """Push a model folder to the HuggingFace Hub."""
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing model folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload model",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload model",
            )

        click.echo(f"Successfully pushed model to: https://huggingface.co/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1) from e


@services_hf.command("push-dataset")
@click.argument("path")
@click.argument("repo_id")
@click.option(
    "--private/--public", default=False, help="Make the repository private (default: public)"
)
@click.option("--commit-message", default=None, help="Custom commit message for the push")
def hf_push_dataset(path: str, repo_id: str, private: bool, commit_message: str):
    """Push a dataset folder to the HuggingFace Hub."""
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing dataset folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Upload dataset",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Upload dataset",
            )

        click.echo(f"Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1) from e


# =============================================================================
# Xeno-canto subgroup
# =============================================================================


@services.group("xc")
def services_xc():
    """Xeno-canto bird recording database."""
    pass


@services_xc.command("search")
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
def xc_search(species, genus, country, quality, sound_type, max_results, output_format):
    """Search Xeno-canto for bird recordings."""
    import json as json_lib

    from bioamla.core.services import xeno_canto

    try:
        results = xeno_canto.search(
            species=species,
            genus=genus,
            country=country,
            quality=quality,
            sound_type=sound_type,
            max_results=max_results,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1) from e

    if not results:
        click.echo("No recordings found.")
        return

    if output_format == "json":
        click.echo(json_lib.dumps([r.to_dict() for r in results], indent=2))
    elif output_format == "csv":
        import csv
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    else:
        click.echo(f"Found {len(results)} recordings:\n")
        for r in results:
            click.echo(f"XC{r.id}: {r.scientific_name} ({r.common_name})")
            click.echo(f"  Quality: {r.quality} | Type: {r.sound_type} | Length: {r.length}")
            click.echo(f"  Location: {r.location}, {r.country}")
            click.echo(f"  Recordist: {r.recordist}")
            click.echo(f"  URL: {r.url}")
            click.echo()


@services_xc.command("download")
@click.option("--species", "-s", help="Species name (scientific or common)")
@click.option("--genus", "-g", help="Genus name")
@click.option("--country", "-c", help="Country name")
@click.option("--quality", "-q", default="A", help="Recording quality filter (default: A)")
@click.option("--max-recordings", "-n", default=10, type=int, help="Maximum recordings to download")
@click.option("--output-dir", "-o", default="./xc_recordings", help="Output directory")
@click.option("--delay", default=1.0, type=float, help="Delay between downloads in seconds")
def xc_download(species, genus, country, quality, max_recordings, output_dir, delay):
    """Download recordings from Xeno-canto."""
    from bioamla.core.services import xeno_canto

    click.echo("Searching Xeno-canto...")

    try:
        results = xeno_canto.search(
            species=species,
            genus=genus,
            country=country,
            quality=quality,
            max_results=max_recordings,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1) from e

    if not results:
        click.echo("No recordings found.")
        return

    click.echo(f"Found {len(results)} recordings. Starting download...")

    stats = xeno_canto.download_recordings(
        results,
        output_dir=output_dir,
        delay=delay,
        verbose=True,
    )

    click.echo(f"\nDownload complete: {stats['downloaded']}/{stats['total']} recordings")


# =============================================================================
# Macaulay Library subgroup
# =============================================================================


@services.group("ml")
def services_ml():
    """Macaulay Library audio recordings database."""
    pass


@services_ml.command("search")
@click.option("--species-code", "-s", help="eBird species code (e.g., amerob)")
@click.option("--scientific-name", help="Scientific name")
@click.option("--region", "-r", help="Region code (e.g., US-NY)")
@click.option("--min-rating", default=0, type=int, help="Minimum quality rating (1-5)")
@click.option("--max-results", "-n", default=10, type=int, help="Maximum results")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def ml_search(species_code, scientific_name, region, min_rating, max_results, output_format):
    """Search Macaulay Library for audio recordings."""
    import json as json_lib

    from bioamla.core.services import macaulay

    try:
        results = macaulay.search(
            species_code=species_code,
            scientific_name=scientific_name,
            region=region,
            media_type="audio",
            min_rating=min_rating,
            count=max_results,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1) from e

    if not results:
        click.echo("No recordings found.")
        return

    if output_format == "json":
        click.echo(json_lib.dumps([a.to_dict() for a in results], indent=2))
    else:
        click.echo(f"Found {len(results)} recordings:\n")
        for a in results:
            click.echo(f"ML{a.catalog_id}: {a.scientific_name} ({a.common_name})")
            click.echo(f"  Rating: {a.rating}/5 | Duration: {a.duration or 'N/A'}s")
            click.echo(f"  Location: {a.location}, {a.country}")
            click.echo(f"  Contributor: {a.user_display_name}")
            click.echo()


@services_ml.command("download")
@click.option("--species-code", "-s", help="eBird species code (e.g., amerob)")
@click.option("--scientific-name", help="Scientific name")
@click.option("--region", "-r", help="Region code (e.g., US-NY)")
@click.option("--min-rating", default=3, type=int, help="Minimum quality rating (default: 3)")
@click.option("--max-recordings", "-n", default=10, type=int, help="Maximum recordings to download")
@click.option("--output-dir", "-o", default="./ml_recordings", help="Output directory")
def ml_download(species_code, scientific_name, region, min_rating, max_recordings, output_dir):
    """Download recordings from Macaulay Library."""
    from bioamla.core.services import macaulay

    click.echo("Searching Macaulay Library...")

    try:
        results = macaulay.search(
            species_code=species_code,
            scientific_name=scientific_name,
            region=region,
            media_type="audio",
            min_rating=min_rating,
            count=max_recordings,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1) from e

    if not results:
        click.echo("No recordings found.")
        return

    click.echo(f"Found {len(results)} recordings. Starting download...")

    stats = macaulay.download_assets(
        results,
        output_dir=output_dir,
        verbose=True,
    )

    click.echo(f"\nDownload complete: {stats['downloaded']}/{stats['total']} recordings")


# =============================================================================
# Species lookup subgroup
# =============================================================================


@services.group("species")
def services_species():
    """Species name lookup and search."""
    pass


@services_species.command("lookup")
@click.argument("name")
@click.option("--to-common", "-c", is_flag=True, help="Convert scientific to common name")
@click.option("--to-scientific", "-s", is_flag=True, help="Convert common to scientific name")
@click.option("--info", "-i", is_flag=True, help="Show full species information")
def species_lookup(name, to_common, to_scientific, info):
    """Look up species names and convert between formats."""
    from bioamla.core.services import species

    if info:
        result = species.get_species_info(name)
        if result:
            click.echo(f"Scientific name: {result.scientific_name}")
            click.echo(f"Common name: {result.common_name}")
            click.echo(f"Species code: {result.species_code}")
            click.echo(f"Family: {result.family}")
            click.echo(f"Order: {result.order}")
            click.echo(f"Source: {result.source}")
        else:
            click.echo(f"Species not found: {name}")
            raise SystemExit(1)
    elif to_common:
        result = species.scientific_to_common(name)
        if result:
            click.echo(result)
        else:
            click.echo(f"No common name found for: {name}")
            raise SystemExit(1)
    elif to_scientific:
        result = species.common_to_scientific(name)
        if result:
            click.echo(result)
        else:
            click.echo(f"No scientific name found for: {name}")
            raise SystemExit(1)
    else:
        info_result = species.get_species_info(name)
        if info_result:
            click.echo(f"{info_result.scientific_name} - {info_result.common_name}")
        else:
            click.echo(f"Species not found: {name}")
            raise SystemExit(1)


@services_species.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Maximum results")
def species_search(query, limit):
    """Fuzzy search for species by name."""
    from bioamla.core.services import species

    results = species.search(query, limit=limit)

    if not results:
        click.echo(f"No species found matching: {query}")
        return

    click.echo(f"Found {len(results)} matching species:\n")
    for r in results:
        score = r["score"] * 100
        click.echo(f"{r['scientific_name']} - {r['common_name']}")
        click.echo(f"  Code: {r['species_code']} | Family: {r['family']} | Match: {score:.0f}%")
        click.echo()


@services.command("clear-cache")
@click.option("--all", "clear_all", is_flag=True, help="Clear all API caches")
@click.option("--xc", is_flag=True, help="Clear Xeno-canto cache")
@click.option("--ml", is_flag=True, help="Clear Macaulay Library cache")
@click.option("--species", is_flag=True, help="Clear species cache")
def clear_cache(clear_all, xc, ml, species):
    """Clear API response caches."""
    total = 0

    if clear_all or xc:
        from bioamla.core.services import xeno_canto

        count = xeno_canto.clear_cache()
        click.echo(f"Cleared {count} Xeno-canto cache entries")
        total += count

    if clear_all or ml:
        from bioamla.core.services import macaulay

        count = macaulay.clear_cache()
        click.echo(f"Cleared {count} Macaulay Library cache entries")
        total += count

    if clear_all or species:
        from bioamla.core.services import species as species_mod

        count = species_mod.clear_cache()
        click.echo(f"Cleared {count} species cache entries")
        total += count

    if not any([clear_all, xc, ml, species]):
        click.echo("No cache specified. Use --all to clear all caches.")
        return

    click.echo(f"\nTotal: {total} cache entries cleared")


# =============================================================================
# eBird subgroup
# =============================================================================


@services.group("ebird")
def services_ebird():
    """eBird bird observation database."""
    pass


@services_ebird.command("validate")
@click.argument("species_code")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option("--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key")
@click.option("--distance", type=float, default=50, help="Search radius in km")
def ebird_validate(species_code: str, lat: float, lng: float, api_key: str, distance: float):
    """Validate if a species is expected at a location."""
    from bioamla.core.services.integrations import EBirdClient

    client = EBirdClient(api_key=api_key)
    result = client.validate_species_for_location(
        species_code=species_code,
        latitude=lat,
        longitude=lng,
        distance_km=distance,
    )

    if result["is_valid"]:
        click.echo(f"✓ {species_code} is expected at this location")
        click.echo(f"  Found {result['nearby_observations']} nearby observations")
        if result["most_recent_observation"]:
            click.echo(f"  Most recent: {result['most_recent_observation']}")
    else:
        click.echo(f"✗ {species_code} not recently observed at this location")
        click.echo(f"  {result['total_species_in_area']} other species observed nearby")


@services_ebird.command("nearby")
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lng", type=float, required=True, help="Longitude")
@click.option("--api-key", envvar="EBIRD_API_KEY", required=True, help="eBird API key")
@click.option("--distance", type=float, default=25, help="Search radius in km")
@click.option("--days", type=int, default=14, help="Days back to search")
@click.option("--limit", type=int, default=20, help="Maximum results")
@click.option("--output", "-o", help="Output CSV file")
def ebird_nearby(
    lat: float, lng: float, api_key: str, distance: float, days: int, limit: int, output: str
):
    """Get recent eBird observations near a location."""
    import csv
    from pathlib import Path

    from bioamla.core.services.integrations import EBirdClient

    client = EBirdClient(api_key=api_key)
    observations = client.get_nearby_observations(
        latitude=lat,
        longitude=lng,
        distance_km=distance,
        back=days,
        max_results=limit,
    )

    click.echo(f"Found {len(observations)} recent observations:")
    for obs in observations[:10]:
        count_str = f" (x{obs.how_many})" if obs.how_many else ""
        click.echo(f"  {obs.common_name}{count_str} - {obs.location_name}")

    if len(observations) > 10:
        click.echo(f"  ... and {len(observations) - 10} more")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f.handle,
                fieldnames=[
                    "species_code",
                    "common_name",
                    "scientific_name",
                    "location_name",
                    "observation_date",
                    "how_many",
                ],
            )
            writer.writeheader()
            for obs in observations:
                writer.writerow(obs.to_dict())
        click.echo(f"Saved to: {output}")


