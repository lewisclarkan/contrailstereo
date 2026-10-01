"""Configuration for contrailstereo"""

from __future__ import annotations

import hashlib
import json
import os 

from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

# ---------- constants ----------

# WGS84 ellipsoid and geostationary orbit radius
R_EQ    = 6378137.0
R_POL   = 6356752.31414
SAT_R   = 42164160.0

# GOES sub-satellite longitudes [deg]
SAT_LON = {
    16: -75.2,
    17: -137.2,
    18: -137.9,
    19: -75.2,
}

# Variants 
MODES = ("snapshot", "track")
PREPS = ("hp", "psf_iso", "psf_minmtf")
GRID_KINDS = ("latlon", "native")
TIME_MODELS = ("nominal", "lut")
REF_SATS = ("east", "west")
ABI_TIME_VERSION = "09042020"

# ---------- helpers ----------

def _coerce(obj):
    """Normalise field values so equal settings hash equally"""
    for f in fields(obj):
        val = getattr(obj, f.name)
        default = (f.default if f.default is not MISSING
                   else f.default_factory() if f.default_factory is not MISSING
                   else None)
        if isinstance(val, list):
            val = tuple(val)
        if (isinstance(default, float) and isinstance(val, int)
                and not isinstance(val, bool)):
            val = float(val)
        if isinstance(default, tuple) and isinstance(val, tuple) and default:
            if isinstance(default[0], float):
                val = tuple(float(x) for x in val)
        object.__setattr__(obj, f.name, val)
 
 
def _check_keys(cls, d, where):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown {where} field(s): {sorted(unknown)}")
    
# ---------- tracking ----------

@dataclass(frozen=True)
class TrackConfig:
    """Multi-frame tracking settings."""
    dts_min: tuple = (-20.0, -10.0, 0.0, 10.0, 20.0)    # frame offsets [min]
    wind_halfwidth_ms: float = 6.0                      # (u, v) search half-width about prior
    wind_step_ms: float = 2.0                           
    wind_prior_sigma_ms: float = 3.0                    # penalty scale pulling toward ERA5
    n_iter: int = 2                                     # height / wind alternations

    def __post_init__(self):
        _coerce(self)

# ---------- config ----------

@dataclass
class StereoConfig:

    # ------ views -----
    sat_east:   int = 16
    sat_west:   int = 17
    channels:   tuple = (13, 15)

    # ----- mode -----
    mode:   str = "snapshot"
    track:  TrackConfig = field(default_factory=TrackConfig)

    # ----- prep (field conditioning) ----
    prep:           str = "hp"          # "hp" / "psf_iso" / "psf_minmtf" / ...
    hp_sigma_km:    float = 8.0         # high-pass Gaussian scale
    psf_factor:     float = 0.45        # footprint sigma / GSD
    psf_dz_km:      float = 0.0         # layer thickness folded into PSF
    psf_inflate:    float = 1.05

    # ----- grid -----
    grid_kind: str = "latlon"       # "latlon" / "native"
    px_km: float = 1.0 
    min_npx: int = 60
    max_npx: int = 750

    # ----- matching -----
    win_km: float = 25.0                # local-correlation window
    h_lo_km: float = 8.0                # height scan
    h_hi_km: float = 16.0
    dh_km: float = 0.25

    # ----- per-pixel quality mask -----
    r_min:      float = 0.35            # peak local correlation floor
    amp_min_k:  float = 0.25            # reference-view feature amplitude [K]
    obs_min_s_eff: float = 0.0

    # ----- scene QC -----
    no_data_s:           float = 120.0      # if time offset is beyond this, no_data (#TODO change this to be lenient in tracking mode)
    min_valid_frac:      float = 0.5        # valid overlap on grid, else coverage
    max_offset_nowind_s: float = 8.0        # uncorrected offset limit: else offset
    wind_resid_max_km:   float = 0.25       # corrected residual limit: else offset

    # ----- timing -----
    time_mode: str= "lut"
    ref_sat: str = "east"
    abi_time_version: str = ABI_TIME_VERSION
    abi_time_file: str = ""

    # ----- winds (in-scan advection) -----
    use_era5: bool = True
    wind_min_offset_s: float = 2.0      # below this, no correction needed
    wind_levels: tuple = (100, 125, 150, 175, 200, 225, 250, 300, 350, 400)
    wind_sigma_base: float = 2.5        # m/s error floor at cruise
    wind_sigma_frac: float = 0.08       # + fraction of local speed

    # ----- destriping (GOES-17) -----
    stripe_nsig: float = 4.0
    stripe_dilate: int = 1
    stripe_mask: bool = False

    # ----- validation -----
    advect_truth: bool = True


    def __post_init__(self):
        if isinstance(self.track, dict):
            _check_keys(TrackConfig, self.track, "track")
            object.__setattr__(self, "track", TrackConfig(**self.track))
        _coerce(self)
        errs = []
        if self.mode not in MODES:
            errs.append(f"mode {self.mode!r} not in {MODES}")
        if self.grid_kind not in GRID_KINDS:
            errs.append(f"grid_kind {self.grid_kind!r} not in {GRID_KINDS}")
        if self.time_model not in TIME_MODELS:
            errs.append(f"time_model {self.time_model!r} not in {TIME_MODELS}")
        if self.ref_sat not in REF_SATS:
            errs.append(f"ref_sat {self.ref_sat!r} not in {REF_SATS}")
        if self.prep not in PREPS:
            errs.append(f"prep {self.prep!r} not in {PREPS}")
        for s in (self.sat_east, self.sat_west):
            if s not in SAT_LON:
                errs.append(f"satellite {s} has no entry in SAT_LON")
        if len(self.channels) != 2 or self.channels[0] == self.channels[1]:
            errs.append(f"channels must be two distinct bands, got {self.channels}")
        if not (0 <= self.h_lo_km < self.h_hi_km):
            errs.append("need 0 <= h_lo_km < h_hi_km")
        if self.dh_km <= 0 or self.px_km <= 0 or self.win_km <= 0:
            errs.append("dh_km, px_km and win_km must be positive")
        if not self.min_npx <= self.max_npx:
            errs.append("need min_npx <= max_npx")
        if errs:
            raise ValueError("invalid StereoConfig: " + "; ".join(errs))

    # ----- identity ----- 
    
    def to_dict(self) -> dict:
        """Plain nested dict (tuples as lists), JSON-ready."""
        return json.loads(json.dumps(asdict(self)))
    
    def config_hash(self) -> str:
        """Short stable hash of every setting, stamped into all outputs."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]
    
    # ----- construction -----

    @classmethod
    def from_dict(cls, d: dict) -> "StereoConfig":
        """Build from a dict"""
        _check_keys(cls, d, "config")
        return cls(**d)
    

    @classmethod 
    def from_file(cls, path) -> "StereoConfig":
        """JSON or TOML"""
        path = Path(path)
        if path.suffix == ".toml":
            import tomllib
            with open(path, "rb") as f:
                d = tomllib.load(f)
        else:
            d = json.loads(path.read_text())
        d.pop("config_hash", None)
        return cls.from_dict(d)
    

    def to_file(self, path) -> Path:
        """Write as JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            dict(self.to_dict(), config_hash=self.config_hash()), indent=2))
        return path


     
    def set(self, **kw) -> "StereoConfig":
        """replace() that also accepts dotted keys, e.g.
        cfg.set(win_km=30, **{"track.dts_min": (-10, 0, 10)})."""
        top = {k: v for k, v in kw.items() if "." not in k}
        sub = {}
        for k, v in kw.items():
            if "." in k:
                head, tail = k.split(".", 1)
                sub.setdefault(head, {})[tail] = v
        for head, vals in sub.items():
            if head != "track":
                raise KeyError(f"no nested config {head!r}")
            _check_keys(TrackConfig, vals, "track")
            top["track"] = replace(self.track, **vals)
        _check_keys(type(self), top, "config")
        return replace(self, **top)
    

DEFAULT = StereoConfig()


def diff(a: StereoConfig, b: StereoConfig) -> list[tuple[str, object, object]]:
    """[(dotted field, value in a, value in b)] for every differing setting."""
    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}."))
            else:
                out[pre + k] = v
        return out
    fa, fb = flat(a.to_dict()), flat(b.to_dict())
    return [(k, fa[k], fb[k]) for k in fa if fa[k] != fb[k]]


def parse_overrides(items, base: StereoConfig = DEFAULT) -> dict:
    """CLI --set strings -> typed kwargs for StereoConfig.set.
 
    'win_km=30'  'channels=14,15'  'use_era5=false'  'track.dts_min=-10,0,10'
    Types follow the current value in `base`.
    """
    cur = {}
    for k, v in base.to_dict().items():
        if isinstance(v, dict):
            cur.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            cur[k] = v
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        k, s = (x.strip() for x in item.split("=", 1))
        if k not in cur:
            raise KeyError(f"unknown config field {k!r}")
        ref = cur[k]
        if isinstance(ref, bool):
            if s.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{k}: expected true/false, got {s!r}")
            out[k] = s.lower() in ("true", "1")
        elif isinstance(ref, list):
            conv = type(ref[0]) if ref else float
            out[k] = tuple(conv(x) for x in s.split(",") if x.strip())
        elif isinstance(ref, int):
            out[k] = int(s)
        elif isinstance(ref, float):
            out[k] = float(s)
        else:
            out[k] = s
    return out

# ----- paths -----

@dataclass(frozen=True)
class Paths:
    goes_cache: Path
    era5_cache: Path
    outputs: Path
    collocations: Optional[Path] = None
    abi_time: Optional[Path] = None


_PATH_FIELDS = ("goes_cache", "era5_cache", "outputs", "collocations", "abi_time")


def _find_root(start: Path) -> Path:
    """Find folder containing pyproject.toml"""
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start


def load_paths(config_file=None) -> Paths:
    """Resolve file locations"""

    cfg_path = Path(config_file or os.environ.get(
        "CONTRAILSTEREO_CONFIG", "~/.config/contrailstereo.toml")).expanduser()
    table = {}
    if cfg_path.exists():
        import tomllib
        with open(cfg_path, "rb") as f:
            table = tomllib.load(f).get("paths", {})
        unknown = set(table) - set(_PATH_FIELDS) - {"root"}
        if unknown:
            raise KeyError(f"{cfg_path}: unknown [paths] key(s) {sorted(unknown)}")
 
    expand = lambda s: Path(os.path.expandvars(str(s))).expanduser()
    root = expand(os.environ.get("CONTRAILSTEREO_ROOT")
                  or table.get("root") or _find_root(Path.cwd()))
    defaults = dict(goes_cache=root / "data" / "goes_cache",
                    era5_cache=root / "data" / "era5_cache",
                    outputs=root / "outputs",
                    collocations=root / "data" / "collocations.nc",
                    abi_time=root / "data" / "abi")
    out = {}
    for name in _PATH_FIELDS:
        env = os.environ.get(f"CONTRAILSTEREO_{name.upper()}")
        out[name] = expand(env or table.get(name) or defaults[name])
    if (not os.environ.get("CONTRAILSTEREO_COLLOCATIONS")
            and "collocations" not in table
            and not out["collocations"].exists()):
        out["collocations"] = None
    return Paths(**out)

