"""Construction validation: build WeatherGFT (GFT) and PI-IAM4VP (IAM4VP) with
the PATCHED kernel, run a dummy forward, verify shapes/no-NaN, and confirm the
trainable-parameter count is UNCHANGED vs the pre-fix code (fix adds only
buffers/constants, no learnable weights)."""
import os,sys,json,importlib.util
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import torch
OUT=os.environ["OUT"]; STAGE=os.path.dirname(OUT)
res={"models":{}}

def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---- FIXED (current working tree): import via package (relative imports need parent) ----
from Models.WeatherGFT import GFT as GFT_cls
from Models.PI_IAM4VP import IAM4VP as IAM4VP_cls
B,T,C,H,W=2,6,69,32,64
# WeatherGFT GFT.forward(x, zquvtw): x=[B,C,H,W]; zquvtw internal init None-> built from x
gft=GFT_cls(hidden_dim=256, physics_part_coef=0.1, channels=69, block_dt=300).eval()
x=torch.randn(B,1,C,H,W)
try:
    with torch.no_grad(): out=gft(x)
    o=out[0] if isinstance(out,(tuple,list)) else out
    res["models"]["WeatherGFT"]={"params":nparams(gft),"forward_out_shape":list(o.shape),
        "nan":bool(torch.isnan(o).any()),"inf":bool(torch.isinf(o).any())}
except Exception as e:
    res["models"]["WeatherGFT"]={"params":nparams(gft),"forward_error":repr(e)[:300]}

# PI-IAM4VP residual-stable config
iam=IAM4VP_cls(T_data=6,C_data=69,H_data=32,W_data=64,use_physics=False,
    use_physics_residual_corrector=True,physics_residual_hidden_channels=128,
    physics_residual_apply_to="upper_air_only",physics_residual_zero_init=True,
    physics_feature_mode="tendency",physics_residual_hybrid_steps=3,
    physics_residual_hybrid_mode="stable_physical",physics_residual_input_space="physical",
    physics_residual_humidity_mode="relative_to_specific",physics_residual_tendency_clip=8.0).eval()
xr=torch.randn(B,T,C,H,W)
yprev=[torch.randn(B,C,H,W) for _ in range(1)]
tt=torch.full((B,),100.0)
def try_forward(m,label):
    for args in [ (xr,[],tt), (xr,yprev,tt) ]:
        try:
            with torch.no_grad(): out=m(*args)
            o=out[0] if isinstance(out,(tuple,list)) else out
            return {"forward_out_shape":list(o.shape),"nan":bool(torch.isnan(o).any()),
                    "inf":bool(torch.isinf(o).any()),"args":len(args)}
        except Exception as e:
            last=repr(e)[:200]
    return {"forward_error":last}
# supply per-channel normalization (real stats from memmap meta) for physical branch
import json as _j, numpy as _np
try:
    _meta=_j.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
    _mean=_np.array(_meta.get("mean") or _meta.get("the_mean"),dtype="float32")
    _std=_np.array(_meta.get("std") or _meta.get("the_std"),dtype="float32")
    if _mean.shape[0]==69:
        iam.set_physics_normalization(torch.from_numpy(_mean),torch.from_numpy(_std))
    else:
        iam.set_physics_normalization(torch.zeros(69),torch.ones(69))
except Exception:
    iam.set_physics_normalization(torch.zeros(69),torch.ones(69))
r_iam=try_forward(iam,"PI-IAM4VP"); r_iam["params"]=nparams(iam)
res["models"]["PI-IAM4VP"]=r_iam

# ---- OLD param counts (from staged pre-fix sources) via fresh import ----
# reload WeatherGFT from staged OLD file to compare param count only
oldp=os.path.join(STAGE,"WeatherGFT_old.py")
if os.path.exists(oldp):
    spec=importlib.util.spec_from_file_location("wg_old",oldp)
    # OLD file imports from package; give it same globals by execing in module ns
    try:
        m=importlib.util.module_from_spec(spec); sys.modules["wg_old"]=m; spec.loader.exec_module(m)
        gft_old=m.GFT(hidden_dim=256,physics_part_coef=0.1,channels=69,block_dt=300).eval()
        res["param_count_check"]={"WeatherGFT_OLD":nparams(gft_old),"WeatherGFT_FIXED":nparams(gft),
            "unchanged":nparams(gft_old)==nparams(gft)}
    except Exception as e:
        res["param_count_check"]={"error":repr(e)[:200]}
json.dump(res,open(OUT,"w"),indent=2)
print(json.dumps(res,indent=2))
print("WROTE",OUT)
