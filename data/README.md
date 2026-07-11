# Data

This project uses real Inside Airbnb listing data (~120K rows across 10 US/international cities).
CSVs are not committed to the repo due to size (~330MB total).

To reproduce:
1. Go to https://insideairbnb.com/get-the-data/
2. Download `listings.csv.gz` for: NYC, Chicago, Seattle, San Francisco, Boston,
   Nashville, Austin, Jersey City, Hollywood FL, Sydney
3. Unzip each and place in this `data/` folder, named:
   nyc_listings.csv, chicago_listings.csv, seattle_listings.csv, sf_listings.csv,
   boston_listings.csv, nashville_listings.csv, austin_listings.csv,
   jerseycity_listings.csv, hollywoodfl_listings.csv, sydney_listings.csv
4. In `src/cli.py`, confirm `DATA_MODE = "real"` and `CSV_PATHS` lists all 10 files above.

Alternatively, leave `DATA_MODE = "synthetic"` to auto-generate 100K+ realistic
listings without downloading anything.
