# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
Collection of paths used to generate plots. Simply copy one of the into tools/plot_l2_col_rate.py 

(not the best way to manage it though)
"""

### VAD - Reproduce ### 
    # Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra",
        "/home/hao1rng/core/VAD/exps/eval/orig/vad-orig",
        "/home/hao1rng/core/VAD/exps/eval/orig/vlp"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig": "VAD",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra": "VAD-VLP (ALP+SLP)",
        "/home/hao1rng/core/VAD/exps/eval/orig/vad-orig": "VAD*",
        "/home/hao1rng/core/VAD/exps/eval/orig/vlp": "VAD-VLP (ALP+SLP)*",
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # or None

    # Output & style
    save_dir = Path("./exps/plots") / "reproduce_vad_vlp"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### VLP - Encoder Ablations ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp_slp/mlp_enc_full/contra"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig": "VAD",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra": "VAD-VLP (LM-Enc)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp_slp/mlp_enc_full/contra": "VAD-VLP (MLP-Enc)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/ablations") / "vlp_encoder_ablation"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### VLP - Loss Ablations ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra",
        "/home/hao1rng/core/VAD/exps/eval/orig/vad-orig",
        "/home/hao1rng/core/VAD/exps/eval/orig/vlp"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig": "VAD",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra": "VAD-VLP (ALP+SLP)",
        "/home/hao1rng/core/VAD/exps/eval/orig/vad-orig": "VAD*",
        "/home/hao1rng/core/VAD/exps/eval/orig/vlp": "VAD-VLP (ALP+SLP)*",
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # or None

    # Output & style
    save_dir = Path("./exps/plots") / "reproduce_vad_vlp"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### VLP - Encoder Ablations ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/mse",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/contra": "VAD-VLP (Contra Loss)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/alp_slp/llm_enc/mse": "VAD-VLP (MSE Loss)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/ablations") / "vlp_loss_ablation"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### VLP - Temproal Ablations ###
### VLP - Encoder Ablations ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/fut-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-fut-ego_4ts",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-fut-ego_8ts"        
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-ego": "VAD-SLP (Past Ego)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/fut-ego": "VAD-SLP (Future Ego)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-fut-ego_4ts": "VAD-SLP (Past & Future Ego - 4 ts)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/slp/llm_enc/contra/past-fut-ego_8ts": "VAD-SLP (Past & Future Ego - 8 ts)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/ablations") / "vlp_temporal_ablation"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Deterministic ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/latent_given_xy/query_refiner_mlp/mse",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/full_posterior_deterministic/query_refiner_mlp/mse",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-VLP (SLP, Orig. Setup)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/latent_given_xy/query_refiner_mlp/mse": "VAD-VLP (SLP, Extended GT-Encoder)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/full_posterior_deterministic/query_refiner_mlp/mse": "VAD-VLP (SLP, Determetistic Posterior)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "deterministic"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Probabilistic ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/baseline/indep_mu_var",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-VLP (SLP, Orig. Setup)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/baseline/indep_mu_var": "VAD-VLP (SLP, Prob. Baseline - No GT Injection)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1": "VAD-VLP (SLP, Prob. Setup)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-VLP (SLP, Prob. Setup - Improved Training)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "probabilistic"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Probabilistic Sampling ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1_freezepostgrad",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s6/w1e-1_freezepostgrad",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP Deterministic",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-SLP Prob. (S = 1)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad": "VAD-SLP Prob. (S = 2)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1_freezepostgrad": "VAD-SLP Prob. (S = 4)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s6/w1e-1_freezepostgrad": "VAD-SLP Prob. (S = 6)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "probabilistic_sampling"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Probabilistic Sampling - scaling ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-VLP (SLP, Orig. Setup)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad": "VAD-VLP (SLP, Prob. Setup - 2 Samples)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6": "VAD-VLP (SLP, Prob. Setup - Scaled Up)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "probabilistic_sampling-scaling"
    base_filename = "metrics_panel"
    base_fontsize = 16


##############################
### VLP ALP, SLP module ablations ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp_slp/llm_enc/contra",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp/llm_enc/contra",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig": "VAD",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp_slp/llm_enc/contra": "VAD-VLP (ALP + SLP)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/alp/llm_enc/contra": "VAD-VLP (ALP)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-VLP (SLP)",   
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_modules_ablation") 
    base_filename = "metrics_panel"
    base_fontsize = 16

##############################
### PROBABILISTIC MAIN TABLE ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad",
        #"/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1/4gpus_b6",
        #"/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/home_fs_new/exps/eval/vad/orig": "VAD",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP Deterministic",   
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-SLP Probabilistic (S = 1)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad": "VAD-SLP Probabilistic (S = 2)",
        #"/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s4/w1e-1/4gpus_b6": "VAD-VAD-SLP Probabilistic (increased batch size) - s4",
        #"/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/4gpus_b6 ": "VAD-VAD-SLP Probabilistic (increased batch size)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "probabilistic_main_table-v2"
    base_filename = "metrics_panel"
    base_fontsize = 16

##############################
### Det to Prob architecture search ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/latent_given_xy/query_refiner_mlp/mse",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/full_posterior_deterministic/query_refiner_mlp/mse",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/latent_given_xy/query_refiner_mlp/mse": "VAD-SLP (Ext. Enc.)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/full_posterior_deterministic/query_refiner_mlp/mse": "VAD-SLP (Ext. Enc. + Post. Feat.)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/vlp_extension") / "deterministic"
    base_filename = "metrics_panel"
    base_fontsize = 16

##############################
### Test three trajectories setup ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/3tr",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/full_posterior_deterministic/query_refiner_mlp/mse",
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-SLP Probabilistic (1 traj)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/3tr": "VAD-SLP Probabilistic (3 traj)",
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/misc") / "3tr"
    base_filename = "metrics_panel"
    base_fontsize = 16

##############################
### Not needed: setup with injecting nav command ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad_cmd",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad_cmd": "VAD-SLP Probabilistic (with nav cmd)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-SLP Probabilistic (no nav cmd)"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/misc") / "cmd"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Not needed: corrected setup with 3 trajectories training ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/3tr_v2",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad_3tr",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad"
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/3tr_v2": "VAD-SLP Probabilistic (3 traj v2)",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad_3tr": "VAD-SLP Probabilistic - s2 (3 traj v2)",
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/2gpus": "VAD-SLP Probabilistic",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad": "VAD-SLP Probabilistic - s2"
    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = [5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/misc") / "3trj"
    base_filename = "metrics_panel"
    base_fontsize = 16

###############################
### Not needed: corrected setup with 3 trajectories training vs posterior ###
# Experiments to visualize
    exp_paths = [
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad_3tr",
        
    ]

    legend_overrides: Dict[str, str] = {
        "/home/hao1rng/core/VAD/exps/eval/vad-fixed/slp/llm_enc/contra/6ts_orig_fut-ego": "VAD-SLP",
        "/home/hao1rng/core/home_fs_new/exps/eval/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s2/w1e-1_freezepostgrad_3tr": "VAD-SLP Probabilistic - s2 (3 traj v2)",

    }

    exclude_runs_map: Dict[str, List[str]] = {
        # "path_substring": ["run_substring_to_exclude_1", "run_substring_to_exclude_2"]
    }

    # Optional: restrict epochs to plot (otherwise all found epochs)
    epochs_to_plot: Optional[List[int]] = None #[5, 10, 15, 20, 25, 30, 35]  # or None

    # Output & style
    save_dir = Path("./exps/plots/misc") / "3trj"
    base_filename = "metrics_panel"
    base_fontsize = 16