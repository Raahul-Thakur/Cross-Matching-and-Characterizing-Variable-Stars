"""Single-command entrypoint for the Gaia/ASAS-SN variable-star pipeline.

Run:
    python CMCVA.py

Optional arguments are the same as run_pipeline.py, for example:
    python CMCVA.py --sample-size 20000 --max-lightcurves 2000
"""

from run_pipeline import main


if __name__ == "__main__":
    main()
