from experiments.complex_geometry.cylinder_geometry import prepare_geometry, scale

domain, hole = prepare_geometry()
decomposition_config = {
    "polygon_vertices": domain,
    "window_fn_type": "quintic",
    "holes": [hole],
    "block_size": (0.6, 0.6),
    "kappa": 0.2,
    "bbox_left": (0, 0),
    "bbox_right": (1, 1),
    "points_per_block": 1333,
    "eps": 1e-6,
    "eps_full": 1e-3,
    "block_scales": [1.69e-05, 0.003505, 0.00129],
    "block_shift": [-7.9e-06, 0.0072, -1.9e-06],
}

decomposition_config["bbox_left"] = (
    -decomposition_config["block_size"][0] / 4,
    -decomposition_config["block_size"][1] / 4,
)

decomposition_config["bbox_right"] = (
    2000 * scale + decomposition_config["block_size"][0] / 4,
    1200 * scale + decomposition_config["block_size"][1] / 4,
)
