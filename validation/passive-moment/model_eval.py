import math

import numpy as np
import pandas as pd
import opensim as osim

# OpenSim Coordinate.MotionType 1 = Rotational, 2 = Translational, 3 = Coupled
_ROTATIONAL = 1

# Canonical hip degrees-of-freedom: (coordinate name, short title, signed-axis label).
# The sign labels match the model's own convention (verified: hip_flexion + = flexion,
# hip_adduction + = adduction, hip_rotation + = internal rotation).
HIP_DOFS = [
    ("hip_flexion_r",   "Hip flexion",   "ext (-)   flex (+)"),
    ("hip_adduction_r", "Hip adduction", "abd (-)   add (+)"),
    ("hip_rotation_r",  "Hip rotation",  "ext-rot (-)   int-rot (+)"),
]


COORD_LABELS = {
    "hip_flexion_r":   ("hip flexion",   "ext (-)   flex (+)"),
    "hip_adduction_r": ("hip adduction", "abd (-)   add (+)"),
    "hip_rotation_r":  ("hip rotation",  "ext-rot (-)   int-rot (+)"),
    "knee_angle_r":    ("knee flexion",  "ext (0)   flex (+)"),
    "ankle_angle_r":   ("ankle angle",   "plantar (-)   dorsi (+)"),
}

ENG_ANGLE_TO_COORD = {
    "hip_flexion":   "hip_flexion_r",
    "hip_adduction": "hip_adduction_r",
    "hip_rotation":  "hip_rotation_r",
    "knee_flexion":  "knee_angle_r",
}


def _dof_name(df):
    """Short human name of the coordinate this sweep varied (for plot titles)."""
    c = df["sweep_coord"].iloc[0]
    return COORD_LABELS.get(c, (c, ""))[0]


def _xlabel_for(df, override=None):
    """X-axis label for a sweep DataFrame, from its swept coordinate (or override)."""
    if override is not None:
        return override
    c = df["sweep_coord"].iloc[0]
    name, sign = COORD_LABELS.get(c, (c, ""))
    return f"{name.capitalize()} angle (deg)" + (f"   {sign}" if sign else "")


# Model loading
def load_model(path):
    model = osim.Model(path)
    state = model.initSystem()
    return model, state


def muscle_names(model):
    ms = model.getMuscles()
    return [ms.get(i).getName() for i in range(ms.getSize())]


def as_millard(muscle):
    return osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)


def coordinate_info(model):
    cs = model.getCoordinateSet()
    rows = []
    for i in range(cs.getSize()):
        c = cs.get(i)
        rotational = c.getMotionType() == _ROTATIONAL
        conv = math.degrees if rotational else (lambda x: x)
        rows.append(dict(
            name=c.getName(),
            kind="rotational" if rotational else ("translational" if c.getMotionType() == 2 else "coupled"),
            range_min=round(conv(c.getRangeMin()), 3),
            range_max=round(conv(c.getRangeMax()), 3),
            unit="deg" if rotational else "m",
            default=round(conv(c.getDefaultValue()), 3),
            clamped=c.get_clamped(), locked=c.get_locked(),
        ))
    return pd.DataFrame(rows)


def coord_range_deg(model, coord_name, n=61, override_deg=None):
    c = model.getCoordinateSet().get(coord_name)
    lo, hi = override_deg if override_deg is not None else (
        math.degrees(c.getRangeMin()), math.degrees(c.getRangeMax()))
    return np.linspace(lo, hi, n)


def active_operating_range(model, muscle_name):
    mu = as_millard(model.getMuscles().get(muscle_name))
    afl = mu.get_ActiveForceLengthCurve()
    return (afl.get_min_norm_active_fiber_length(),
            afl.get_transition_norm_fiber_length(),
            afl.get_max_norm_active_fiber_length())


def default_pose(model):
    cs = model.getCoordinateSet()
    return {cs.get(i).getName(): cs.get(i).getDefaultValue() for i in range(cs.getSize())}

def set_pose(model, state, pose):
    """Set every coordinate in `pose` (native units) then assemble() so coupled
    coordinates (knee beta) are solved. Coordinates absent from `pose` are left."""
    cs = model.getCoordinateSet()
    for name, val in pose.items():
        cs.get(name).setValue(state, float(val), False)  # defer constraint solve
    model.assemble(state)  # enforce CoordinateCouplerConstraint(s) once
    return state

def sweep(model, state, sweep_coord, angles_deg,
          muscles=None, base_pose=None, moment_arm_coords=None,
          equilibrate=True, activation=0.0):
    """Sweep one rotational coordinate across `angles_deg`, measuring per muscle.

    Parameters
    ----------
    sweep_coord : str          rotational coordinate name to vary (x-axis).
    angles_deg : iterable       angles in DEGREES.
    muscles : list[str]|None    muscles to measure (default: all in model).
    base_pose : dict|None       native-unit pose for the non-swept coordinates
                                (default: model default pose). Set e.g.
                                {'knee_angle_r': radians(15)} here to hold the knee.
    moment_arm_coords : list[str]|None  coordinates to report moment arms about
                                (default: just `sweep_coord`).
    equilibrate : bool          solve muscle fiber equilibrium at each pose
                                (required for normalized fiber length & passive
                                force; skip for purely kinematic MA/length sweeps).
    activation : float          activation held during equilibration (0 = passive).

    Returns tidy DataFrame, one row per (angle, muscle), with columns:
        muscle, sweep_coord, angle_deg, length_cm, norm_fiber_length,
        pennation_rad, passive_fiber_force_N, passive_force_along_tendon_N,
        moment_arm__<coord>_cm, passive_moment__<coord>_Nm (one pair per MA coord).
    """
    cs = model.getCoordinateSet()
    if muscles is None:
        muscles = muscle_names(model)
    if base_pose is None:
        base_pose = default_pose(model)
    if moment_arm_coords is None:
        moment_arm_coords = [sweep_coord]

    mu_objs = {nm: as_millard(model.getMuscles().get(nm)) for nm in muscles}
    ma_coords = {cn: cs.get(cn) for cn in moment_arm_coords}

    rows = []
    for ang in angles_deg:
        pose = dict(base_pose)
        pose[sweep_coord] = math.radians(ang)
        set_pose(model, state, pose)

        if equilibrate:
            model.realizeVelocity(state)
            for mu in mu_objs.values():
                mu.setActivation(state, activation)
            model.equilibrateMuscles(state)
            model.realizeDynamics(state)
        else:
            model.realizePosition(state)

        for nm, mu in mu_objs.items():
            row = dict(muscle=nm, sweep_coord=sweep_coord, angle_deg=float(ang))
            row["length_cm"] = mu.getLength(state) * 100.0
            if equilibrate:
                pen = mu.getPennationAngle(state)
                Fp = mu.getPassiveFiberForce(state)
                row["norm_fiber_length"] = mu.getNormalizedFiberLength(state)
                row["pennation_rad"] = pen
                row["passive_fiber_force_N"] = Fp
                row["passive_force_along_tendon_N"] = Fp * math.cos(pen)
            for cn, co in ma_coords.items():
                ma = mu.computeMomentArm(state, co)  # metres
                row[f"moment_arm__{cn}_cm"] = ma * 100.0
                if equilibrate:
                    row[f"passive_moment__{cn}_Nm"] = row["passive_force_along_tendon_N"] * ma
            rows.append(row)
    return pd.DataFrame(rows)


def sum_passive_moment(df, coord):
    col = f"passive_moment__{coord}_Nm"
    if col not in df.columns:
        raise KeyError(
            f"{col!r} not in sweep df — run sweep(..., equilibrate=True, "
            f"moment_arm_coords=[..., {coord!r}])")
    return (df.groupby("angle_deg", as_index=False)[col].sum()
              .rename(columns={col: "total_moment_Nm"}))


# Muscle grouping
def classify_muscles(muscles):
    itb = [m for m in muscles if "ITB" in m]
    other = [m for m in muscles if "ITB" not in m]
    groups = {}
    if itb:
        groups["ITB tracts"] = itb
    if other:
        groups["femoral glutes"] = other
    return groups


def muscle_colors(muscles, cmaps=("Reds", "Blues"), lo=0.45, hi=0.9):
    import matplotlib.pyplot as plt
    groups = classify_muscles(muscles)
    colors = {}
    for (label, members), cmap_name in zip(groups.items(), cmaps):
        cmap = plt.get_cmap(cmap_name)
        n = len(members)
        for i, m in enumerate(members):
            frac = hi if n == 1 else lo + (hi - lo) * i / (n - 1)
            colors[m] = cmap(frac)
    return colors


# Two-model compariosn
def series_key(label, muscle):
    return f"{label}: {muscle}"


def sweep_compare(specs, sweep_coord, angles_deg, moment_arm_coords=None,
                  equilibrate=True, activation=0.0):
    frames = []
    for spec in specs:
        df = sweep(spec["model"], spec["state"], sweep_coord, angles_deg,
                   muscles=spec["muscles"], base_pose=spec.get("base_pose"),
                   moment_arm_coords=moment_arm_coords,
                   equilibrate=equilibrate, activation=activation).copy()
        df["model"] = spec["label"]
        df["muscle_name"] = df["muscle"]
        df["muscle"] = [series_key(spec["label"], mu) for mu in df["muscle_name"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def compare_colors(specs, cmaps=("Reds", "Blues"), lo=0.45, hi=0.9):
    import matplotlib.pyplot as plt
    colors = {}
    for spec, cmap_name in zip(specs, cmaps):
        cmap = plt.get_cmap(cmap_name)
        ms = spec["muscles"]
        n = len(ms)
        for i, mu in enumerate(ms):
            frac = hi if n == 1 else lo + (hi - lo) * i / (n - 1)
            colors[series_key(spec["label"], mu)] = cmap(frac)
    return colors

def classify_by_model(specs):
    def _classify(series):
        groups = {}
        present = set(series)
        for spec in specs:
            keys = [series_key(spec["label"], mu) for mu in spec["muscles"]]
            keys = [k for k in keys if k in present]
            if keys:
                groups[spec["label"]] = keys
        return groups
    return _classify


def compare_op_bands(specs, cmaps=("Reds", "Blues")):
    import matplotlib.pyplot as plt
    seen = {}  # (rounded lo, hi) -> dict, insertion-ordered
    for spec, cmap_name in zip(specs, cmaps):
        lo, _plat, hi = active_operating_range(spec["model"], spec["muscles"][0])
        key = (round(lo, 4), round(hi, 4))
        if key in seen:
            seen[key]["labels"].append(spec["label"])
        else:
            seen[key] = {"lo": lo, "hi": hi, "labels": [spec["label"]], "cmap": cmap_name}
    bands = []
    for info in seen.values():
        if len(info["labels"]) > 1:
            color, tag = "0.6", "shared"
        else:
            color, tag = plt.get_cmap(info["cmap"])(0.7), info["labels"][0]
        bands.append((info["lo"], info["hi"], color,
                      f"{tag} active range [{info['lo']:.2f}, {info['hi']:.2f}]"))
    return bands

# Digitized reference moment-arm bands
def load_reference_bands(eng_csv=None, blemker_csv=None):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    dig = os.path.join(here, "..", "eng-curve-digitization")
    eng_csv = eng_csv or os.path.join(dig, "eng-itb-moment-arms-long.csv")
    blemker_csv = blemker_csv or os.path.join(dig, "blemker-gmax-moment-arms-long.csv")

    def _pair(df):  # align upper & lower on the shared per-series grid_idx
        up = df[df.bound_type == "upper"].sort_values("grid_idx")
        lo = df[df.bound_type == "lower"].sort_values("grid_idx")
        return up.angle_deg.values, lo.moment_arm_cm.values, up.moment_arm_cm.values

    bands = {}
    if os.path.exists(eng_csv):
        eng = pd.read_csv(eng_csv)
        for atype, coord in ENG_ANGLE_TO_COORD.items():
            sub = eng[eng.angle_type == atype]
            for tract in sub.model_tract.unique():
                t = sub[sub.model_tract == tract]
                for name in sorted(t.muscle.unique()):
                    ang, lo, hi = _pair(t[t.muscle == name])
                    bands.setdefault(coord, []).append(dict(
                        angle=ang, lower=lo, upper=hi, muscle=tract,
                        source="Eng", label=f"Eng {name}"))
    if os.path.exists(blemker_csv):
        blem = pd.read_csv(blemker_csv)
        for atype in blem.angle_type.unique():
            coord = ENG_ANGLE_TO_COORD.get(atype)
            if coord is None:
                continue
            ang, lo, hi = _pair(blem[blem.angle_type == atype])
            bands.setdefault(coord, []).append(dict(
                angle=ang, lower=lo, upper=hi, muscle=None,
                source="Blemker", label="Blemker gmax"))
    return bands


def _draw_bands(ax, band_list, colors=None):
    sources = []
    for b in band_list:
        c = colors.get(b["muscle"]) if (colors and b.get("muscle") in colors) else "0.45"
        ax.fill_between(b["angle"], b["lower"], b["upper"], color=c, alpha=0.15,
                        lw=0, zorder=-3)
        ax.plot(b["angle"], b["lower"], color=c, lw=0.7, alpha=0.5, zorder=-3)
        ax.plot(b["angle"], b["upper"], color=c, lw=0.7, alpha=0.5, zorder=-3)
        if b["source"] not in sources:
            sources.append(b["source"])
    return sources


# Plots
def _legend_by_group(ax, muscles, colors, classify=None, extra=None, **kw):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    classify = classify or classify_muscles
    ordered = [m for grp in classify(muscles).values() for m in grp]
    handles = [Line2D([0], [0], color=colors[m], lw=2.2, label=m) for m in ordered]
    if extra:
        handles += [Patch(facecolor=c, alpha=0.35, label=l) for l, c in extra]
    ax.legend(handles=handles, fontsize=8, **kw)


def plot_moment_arm_grid(sweeps, muscles, colors=None, dofs=HIP_DOFS,
                         figsize=(13, 11), title=None, ylims=None, classify=None,
                         bands=None):
    import matplotlib.pyplot as plt
    if colors is None:
        colors = muscle_colors(muscles)
    ma_dofs = dofs           # rows
    sweep_dofs = dofs        # columns
    n = len(ma_dofs)
    fig, axes = plt.subplots(n, n, figsize=figsize, sharex="col")

    band_sources = []
    for r, (ma_coord, ma_title, _) in enumerate(ma_dofs):
        for c, (sw_coord, sw_title, sw_axis) in enumerate(sweep_dofs):
            ax = axes[r, c]
            # reference bands only make sense on the diagonal (MA about X vs X swept)
            if bands is not None and ma_coord == sw_coord and ma_coord in bands:
                for s in _draw_bands(ax, bands[ma_coord], colors):
                    if s not in band_sources:
                        band_sources.append(s)
            df = sweeps[sw_coord]
            col = f"moment_arm__{ma_coord}_cm"
            for m in muscles:
                d = df[df.muscle == m]
                ax.plot(d.angle_deg, d[col], color=colors[m], lw=1.8)
            ax.axhline(0, color="0.6", lw=0.8, zorder=0)
            ax.grid(True, alpha=0.25)
            _yl = ylims.get(ma_coord) if isinstance(ylims, dict) else ylims
            if _yl is not None:
                ax.set_ylim(_yl)
            if r == 0:
                ax.set_title(sw_title, fontsize=11, fontweight="bold")
            if r == n - 1:
                ax.set_xlabel(f"{sw_title} angle (deg)\n{sw_axis}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{ma_title}\nmoment arm (cm)", fontsize=9)
    extra = [(f"{s} MA band", "0.45") for s in band_sources] or None
    _legend_by_group(axes[0, n - 1], muscles, colors, classify=classify,
                     extra=extra, loc="best")
    fig.suptitle(title or "Hip moment arms  —  row = moment about,  column = angle swept",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_passive_moment(df, muscles, moment_coord="hip_flexion_r", colors=None,
                        figsize=(8, 5.5), knee_deg=None, title=None, ylim=None,
                        classify=None, xlabel=None):
    import matplotlib.pyplot as plt
    if colors is None:
        colors = muscle_colors(muscles)
    col = f"passive_moment__{moment_coord}_Nm"
    dofname = COORD_LABELS.get(moment_coord, (moment_coord, ""))[0]
    fig, ax = plt.subplots(figsize=figsize)
    for m in muscles:
        d = df[df.muscle == m]
        ax.plot(d.angle_deg, d[col], color=colors[m], lw=2.0, label=m)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(_xlabel_for(df, xlabel))
    ax.set_ylabel("Passive moment (N·m)")
    if ylim is not None:
        ax.set_ylim(ylim)
    sub = f"  (knee held at {knee_deg:g}°)" if knee_deg is not None else ""
    ax.set_title(title or f"Passive {dofname} moment per muscle{sub}", fontweight="bold")
    _legend_by_group(ax, muscles, colors, classify=classify, loc="best")
    fig.tight_layout()
    return fig


def plot_moment_arm(df, muscles, moment_coord, colors=None, figsize=(8, 5.5),
                    title=None, ylim=None, classify=None, xlabel=None, bands=None):
    import matplotlib.pyplot as plt
    if colors is None:
        colors = muscle_colors(muscles)
    col = f"moment_arm__{moment_coord}_cm"
    dofname = COORD_LABELS.get(moment_coord, (moment_coord, ""))[0]
    fig, ax = plt.subplots(figsize=figsize)
    extra = None
    if bands is not None and moment_coord in bands:
        extra = [(f"{s} MA band", "0.45") for s in _draw_bands(ax, bands[moment_coord], colors)]
    for m in muscles:
        d = df[df.muscle == m]
        ax.plot(d.angle_deg, d[col], color=colors[m], lw=2.0, label=m)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(_xlabel_for(df, xlabel))
    ax.set_ylabel(f"{dofname.capitalize()} moment arm (cm)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title or f"{dofname.capitalize()} moment arm vs {_dof_name(df)}", fontweight="bold")
    _legend_by_group(ax, muscles, colors, classify=classify, extra=extra, loc="best")
    fig.tight_layout()
    return fig


def plot_band_validation_grid(model, state, panels, bands=None, colors=None,
                              n=61, ncols=3, figsize=None, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    colors = colors or {}
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (5.3 * ncols, 4.3 * nrows),
                             squeeze=False)
    for idx, p in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        coord, muscles = p["coord"], p["muscles"]
        angles = coord_range_deg(model, coord, n, override_deg=p.get("range_deg"))
        df = sweep(model, state, coord, angles, muscles=muscles,
                   moment_arm_coords=[coord], equilibrate=False)
        col = f"moment_arm__{coord}_cm"
        srcs = []
        if bands is not None and p.get("band_source") and coord in bands:
            want = p["band_source"]
            panel_bands = [b for b in bands[coord] if b["source"] == want
                           and (b["muscle"] is None or b["muscle"] in muscles)]
            srcs = _draw_bands(ax, panel_bands, colors)
        for m in muscles:
            d = df[df.muscle == m]
            ax.plot(d.angle_deg, d[col], color=colors.get(m, "k"), lw=2.0, label=m)
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.grid(True, alpha=0.25)
        name, sign = COORD_LABELS.get(coord, (coord, ""))
        ax.set_title(p.get("label", coord), fontsize=9.5, fontweight="bold")
        ax.set_xlabel(f"{name} angle (deg)" + (f"   {sign}" if sign else ""), fontsize=8.5)
        ax.set_ylabel("moment arm (cm)", fontsize=8.5)
        handles = [Line2D([0], [0], color=colors.get(m, "k"), lw=2.2, label=m) for m in muscles]
        handles += [Patch(facecolor="0.45", alpha=0.35, label=f"{s} band") for s in srcs]
        ax.legend(handles=handles, fontsize=7.5, loc="best")
    for j in range(len(panels), nrows * ncols):        # hide unused cells
        axes[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    return fig


def plot_norm_fiber_length(df, muscles, model=None, operating_range=None,
                           colors=None, figsize=(8, 5.5), title=None, ylim=None,
                           classify=None, op_bands=None, xlabel=None):
    import matplotlib.pyplot as plt
    if colors is None:
        colors = muscle_colors(muscles)
    fig, ax = plt.subplots(figsize=figsize)
    if op_bands is not None:
        for lo, hi, color, label in op_bands:
            ax.axhspan(lo, hi, color=color, alpha=0.14, zorder=0, label=label)
    else:
        if operating_range is None and model is not None:
            operating_range = active_operating_range(model, muscles[0])
        if operating_range is not None:
            lo, plat, hi = operating_range
            ax.axhspan(lo, hi, color="0.75", alpha=0.30, zorder=0,
                       label=f"active range [{lo:.2f}, {hi:.2f}]")
            ax.axhspan(plat, 1.0, color="0.55", alpha=0.30, zorder=0)  # plateau
    ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=1, label="optimal (L=1)")
    for m in muscles:
        d = df[df.muscle == m]
        ax.plot(d.angle_deg, d.norm_fiber_length, color=colors[m], lw=2.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(_xlabel_for(df, xlabel))
    ax.set_ylabel("Normalized fiber length  (L / L₀)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title or f"Fiber operating length vs {_dof_name(df)}", fontweight="bold")
    _legend_by_group(ax, muscles, colors, classify=classify, loc="best")
    fig.tight_layout()
    return fig


def plot_mt_length(df, muscles, colors=None, figsize=(8, 5.5), title=None, ylim=None,
                   classify=None, xlabel=None):
    import matplotlib.pyplot as plt
    if colors is None:
        colors = muscle_colors(muscles)
    fig, ax = plt.subplots(figsize=figsize)
    for m in muscles:
        d = df[df.muscle == m]
        ax.plot(d.angle_deg, d.length_cm, color=colors[m], lw=2.0, label=m)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(_xlabel_for(df, xlabel))
    ax.set_ylabel("Muscle–tendon length (cm)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title or f"Muscle–tendon length vs {_dof_name(df)}  (wrap-glitch detector)",
                 fontweight="bold")
    _legend_by_group(ax, muscles, colors, classify=classify, loc="best")
    fig.tight_layout()
    return fig
