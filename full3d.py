# full3d.py (your code)
def simulate(inputs, progress=None):
    """
    inputs: dict with keys:
      grid, rock (kx_md, ky_md, phi, Ti_mult, Tj_mult), pvt, relperm, schedule,
      msw (laterals, L_ft, stage_spacing_ft, clusters_per_stage, dp_limited_entry_psi, friction_factor, well_ID_ft, xf_ft, hf_ft, weights),
      stress (CfD0, alpha_sigma, sigma_overburden_psi, refrac_day, refrac_recovery),
      init (p_init_psi, pwf_min_psi, Sw_init), include_rs_in_mb (bool)
    progress: optional callback like progress("assembling A", 0.3)

    return dict(t_days=..., qg_Mscfd=..., qo_STBpd=..., p3d_psi=..., pf_mid_psi=..., pm_mid_psi=..., Sw_mid=...)
    """
    # ... your Newton/implicit loop here ...
    return results
