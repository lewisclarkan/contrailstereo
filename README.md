# Contrailstereo

Height retrieval for contrails and thin cirrus from geostationary stereo. Since two geostationary satellites see the same cloud from different angles, its apparent position differs between the two images by an amount that depends on its height. For each hypothesised height, the two views are projected onto a common grid and correlated locally. The height that best aligns them is the retrieved height for each pixel.

Validation is against CALIOP lidar cloud tops collocated with GOES contrail detections (Meijer 2024).

## Quick start

```bash
# resolved config, its hash, and the paths in use
contrailstereo config
 
# default configuration over the dev split: verdicts, summary, gate ladder
contrailstereo run --split dev
 
# A/B test a change on the same cases, with paired confidence intervals
contrailstereo compare default default --set-b prep=psf_iso \
    --log --note "PSF iso vs hp"
 
# a single scene with no truth (e.g. a trial window); saves the height map
contrailstereo run --time "2019-12-01 21:12" --bbox 32.8 34.8 -116 -114
```
 
Runs are cached by configuration hash under `<outputs>/runs`: `run` resumes
after an interruption, and `compare` reuses anything already computed.
`notebooks/default_run.ipynb` does the same as `run` with figures.

## Repository layout
 
```
src/contrailstereo/
  config.py       StereoConfig (hashed), Paths (not hashed), constants
  types.py        Case, Grid, Frame, WindProfile, Result
  geometry.py     parallax, viewing geometry, grids, locate
  data/
    goes.py       ABI fetch + cache, destriping
    era5.py       wind profiles
    caliop.py     collocation file -> Cases
  prep.py         BTD sampling, high-pass, PSF homogenisation
  retrieve.py     Case -> Result
  validation.py   runs, truth, metrics, A/B comparison, legacy import
  vis.py          figures
  cli.py          contrailstereo run / compare / config
notebooks/
  default_run.ipynb
scripts/
  check_v4.py     reproduce the v0.1 headline numbers
archive/          exploratory notebooks from v0.1, not maintained
```

## Outputs
 
```
<outputs>/runs/
  {name}_{hash}_config.json     full configuration
  {name}_{hash}_cases.jsonl     one line per case: verdict and diagnostics
  {name}_{hash}_profiles.csv    one row per truth point: top_km, h, r
  {name}_{hash}_maps/           per-case height/r/amp netCDF (--maps)
<outputs>/experiments/log.csv   compare --log history
```
 
Editing a variant's configuration changes its hash, so it starts new files
rather than mixing results. Pipeline failures are recorded as
`qc = "error"` with the message, and retried on the next run.

## Status and limitations
 
- **Snapshot mode only.** `mode = "track"` (multi-frame matching with a
  fitted wind) is defined in the configuration and raises
  `NotImplementedError`.
- **Lat/lon grid only.** `grid_kind = "native"` (matching on a satellite's
  own pixel grid, avoiding resampling of the reference view) is planned.
- **Not yet contrail-aware.** The retrieval matches any texture in the
  BTD field. Planned options: detection masks as matching support,
  height consistency along a contrail, physical (flight-level / ISSR)
  height priors, orientation gating against the disparity direction.
- **GOES-East/West over North America.** Other pairs need their
  sub-satellite longitudes in `SAT_LON` and a data module.
- **No tests in this release.** CI runs an import and CLI smoke check;
  the test suite is being rewritten.