
from pathlib import Path
import shutil


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR = RESULTS_DIR / "models"
GRAPHS_DIR = RESULTS_DIR / "graphs"
HISTORY_DIR = RESULTS_DIR / "history"


def create_directories():
    """Create the organized results directories."""
    for directory in [
        MODELS_DIR,
        GRAPHS_DIR,
        HISTORY_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def move_files():
    """Move checkpoints, graphs, and history files."""

    file_destinations = {
        ".pth": MODELS_DIR,
        ".png": GRAPHS_DIR,
        ".json": HISTORY_DIR,
    }

    moved_count = 0

    # Only process files directly inside results/
    for file_path in list(RESULTS_DIR.iterdir()):

        if not file_path.is_file():
            continue

        destination_dir = file_destinations.get(
            file_path.suffix.lower()
        )

        if destination_dir is None:
            continue

        destination_path = destination_dir / file_path.name

        # Never overwrite an existing file
        if destination_path.exists():
            print(f"Skipped (already exists): {file_path.name}")
            continue

        shutil.move(
            str(file_path),
            str(destination_path),
        )

        print(f"Moved: {file_path.name} -> {destination_dir.name}/")
        moved_count += 1

    print(f"\nOrganization complete. Files moved: {moved_count}")


if __name__ == "__main__":
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}"
        )

    create_directories()
    move_files()
