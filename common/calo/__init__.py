"""calo — shared toy-calorimeter + flow-matching toolkit for the Aug projects."""
from .cfm import CFMTrainer, ema_from, euler_sample, make_pairs, update_ema
from .data import (
    Scaler,
    get_data,
    lateral_profiles,
    layer_profiles,
    normalize_images,
    physics_features,
    simulate,
    split_data,
    summary_vector,
)
from .metrics import (
    auc_score,
    c2st_auc,
    compare_observables,
    coverage,
    permutation_test,
    radial_moment,
    wasserstein1,
)
from .nets import AttnUNet, DeepSets, DiT, Discriminator, RealNVP, ResMLP, VAE
from .util import (
    REPO_ROOT,
    Timer,
    count_params,
    fmt_table,
    hash_params,
    human_int,
    save_json,
    seed_all,
)
from .viz import (
    bar_plot,
    compare_grid,
    heatmap,
    hist_compare,
    image_grid,
    line_plot,
    loss_curve,
    profile_compare,
    save,
    scatter3d,
)

__all__ = [name for name in dir() if not name.startswith("_")]
__version__ = "0.1.0"
