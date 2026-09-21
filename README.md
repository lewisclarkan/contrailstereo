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
  test.py     reproduce the v0.1 headline numbers
```
