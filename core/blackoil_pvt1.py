@dataclass
class BlackOilPVT:
    pb_psi: float
    Rs_pb_scf_stb: float
    Bo_pb_rb_stb: float
    muo_pb_cp: float
    mug_pb_cp: float
    ct_o_1psi: float = 8e-6
    ct_g_1psi: float = 3e-6
    ct_w_1psi: float = 3e-6
    Bw_ref: float = 1.01  # simple constant water FVF

    # --- REPLACE your existing from_inputs with this one ---
    @classmethod
    def from_inputs(cls, data) -> "BlackOilPVT":
        """
        Accepts:
          - dict in the new format  -> build BlackOilPVT
          - an existing BlackOilPVT -> return as-is
          - legacy Fluid-like object -> adapt to dict and build
        """
        # already constructed?
        if isinstance(data, cls):
            return data

        # new-format dict?
        if isinstance(data, dict):
            d = data
        else:
            # legacy object: class named 'Fluid' or with tell-tale fields
            name = type(data).__name__.lower()
            if (name == "fluid"
                or hasattr(data, "pb_psi")
                or hasattr(data, "Rs_pb_scf_stb")
                or hasattr(data, "Bo_pb_rb_stb")):
                d = cls._legacy_to_dict(data)
            else:
                # last-resort: mapping-like
                try:
                    d = dict(data)
                except Exception as e:
                    raise TypeError(f"Unsupported PVT input type for BlackOilPVT: {type(data)}") from e

        # build with robust defaults
        return cls(
            pb_psi=float(d.get("pb_psi", 5200.0)),
            Rs_pb_scf_stb=float(d.get("Rs_pb_scf_stb", 650.0)),
            Bo_pb_rb_stb=float(d.get("Bo_pb_rb_stb", 1.35)),
            muo_pb_cp=float(d.get("muo_pb_cp", 1.2)),
            mug_pb_cp=float(d.get("mug_pb_cp", 0.020)),
            ct_o_1psi=float(d.get("ct_o_1psi", 8e-6)),
            ct_g_1psi=float(d.get("ct_g_1psi", 3e-6)),
            ct_w_1psi=float(d.get("ct_w_1psi", 3e-6)),
            Bw_ref=float(d.get("Bw_ref", 1.01)),
        )

    # --- ADD this helper right below from_inputs ---
    @staticmethod
    def _legacy_to_dict(obj) -> dict:
        """Map a legacy Fluid-like object to the dict this class expects."""
        def pick(*names, default=None):
            for nm in names:
                if hasattr(obj, nm):
                    return getattr(obj, nm)
                d = getattr(obj, "__dict__", None)
                if isinstance(d, dict) and nm in d:
                    return d[nm]
            return default

        return {
            # primary pressure & solution gas
            "pb_psi":         pick("pb_psi", "pb", "p_b", default=5200.0),
            "Rs_pb_scf_stb":  pick("Rs_pb_scf_stb", "Rs_pb", "Rs_pb_stb", "Rs_pb_scf", "Rs", default=650.0),
            # FVF at pb
            "Bo_pb_rb_stb":   pick("Bo_pb_rb_stb", "Bo_pb", "Bo", default=1.35),
            # viscosities (cP)
            "muo_pb_cp":      pick("muo_pb_cp", "mu_oil_cP", "muo_pb", "mu_o", default=1.2),
            "mug_pb_cp":      pick("mug_pb_cp", "mu_gas_cP", "mu_g", default=0.020),
            # compressibilities (1/psi)
            "ct_o_1psi":      pick("ct_o_1psi", "co_1psi", "c_oil_1psi", default=8e-6),
            "ct_g_1psi":      pick("ct_g_1psi", "cg_1psi", "c_gas_1psi", default=3e-6),
            "ct_w_1psi":      pick("ct_w_1psi", "cw_1psi", "c_water_1psi", default=3e-6),
            # water FVF reference
            "Bw_ref":         pick("Bw_ref", "Bw", default=1.01),
        }
