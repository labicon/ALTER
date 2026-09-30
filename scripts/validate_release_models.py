#!/usr/bin/env python3
"""CPU loading/inference and optional one-step training checks for selected models."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.image_diffusion import ImageConditional_ODE
from src.image_coordination import ImageCoordinationHead, LEGACY_DECODER_EXECUTION, side_net_kwargs_from_stats, plan_memory_kwargs_from_stats


def make_policy(s):
    return ImageConditional_ODE(x_dim=7, sigma_data=float(s.get('sigma_data',1)), d_model=int(s['d_model']), n_heads=int(s['n_heads']), depth=int(s['depth']), dim_feedforward=int(s['dim_feedforward']), horizon=int(s['horizon']), device='cpu', num_cameras=int(s['num_cameras']), backbone=s.get('backbone','resnet18'), frame_offsets=s.get('frame_offsets',[0]))


def state_hash(model):
    h=hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,default=Path('documentation/release/workflow-artifacts.json'))
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(123)
    rows=json.loads(args.matrix.read_text());base_row=next(r for r in rows if r['label']=='Frozen forward-only base')
    def stats(row):return pickle.loads((args.inputs/row['stats']).read_bytes())
    bs=stats(base_row);base=make_policy(bs);assert base.load(args.inputs/base_row['checkpoint'])
    for network in [base.F,base.F_ema]:network.requires_grad_(False).eval()
    frozen=(state_hash(base.F),state_hash(base.F_ema))
    image=torch.linspace(0,1,3*128*128).reshape(1,3,128,128);x=torch.linspace(-1,1,140).reshape(1,20,7);sigma=torch.ones(1,1,1)
    reports=[]
    for row in rows:
        torch.manual_seed(123);s=stats(row);coord=row['kind'] in ('coord', 'capacity-coord')
        if coord:
            def construct():
                return ImageCoordinationHead(x_dim=7,base_d_model=s.get('base_d_model',256),d_model=s['head_d_model'],n_heads=s['head_n_heads'],depth=s['head_depth'],dim_feedforward=s['head_dim_feedforward'],horizon=bs['horizon'],sigma_data=bs['sigma_data'],num_cameras=bs['num_cameras'],**side_net_kwargs_from_stats(s,base.F.tokens_per_camera),decoder_execution=s.get('decoder_execution',LEGACY_DECODER_EXECUTION),decoder_conditioning=s.get('decoder_conditioning','pooled'),**plan_memory_kwargs_from_stats(s),frame_offsets=bs.get('frame_offsets',[0]))
            model=construct();assert model.load(args.inputs/row['checkpoint'],device='cpu');model.eval()
            with torch.no_grad():
                enc=base.F_ema.forward_encoder(None,image);d=base._D_from_enc(x,sigma,enc,use_ema=True);value=model.forward_residual(x,sigma,enc,use_ema=True,d_base=d,imgs_shoulder=image)
        else:
            def construct():return make_policy(s)
            model=construct();assert model.load(args.inputs/row['checkpoint']);model.F.eval()
            with torch.no_grad():value=model.sample(None,image,traj_len=20,n_samples=1,N=2)
        assert torch.isfinite(value).all()
        result={'label':row['label'],'loaded':True,'finite_inference':True,'parameters':sum(p.numel() for p in model.F.parameters())}
        if args.training_smoke:
            before=state_hash(model.F);torch.manual_seed(456)
            if coord:
                model.train();loss,gradient,metrics=model.update_mixed((None,image,x),(None,image,x),base)
                assert frozen==(state_hash(base.F),state_hash(base.F_ema));result['frozen_base_unchanged']=True
            else:
                model.F.train();loss,gradient=model.update(x,None,image)
            assert torch.isfinite(torch.tensor(loss));assert before!=state_hash(model.F)
            with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
                saved=Path(temp)/'smoke.pt';model.save(saved);restored=construct()
                if coord:
                    assert restored.load(saved,device='cpu');assert state_hash(restored)==state_hash(model)
                else:
                    assert restored.load(saved);assert state_hash(restored.F)==state_hash(model.F);assert state_hash(restored.F_ema)==state_hash(model.F_ema)
                del restored
            result.update(loss=float(loss),gradient_norm=float(gradient),parameters_updated=True,save_reload_equal=True)
        reports.append(result);print(row['label'],'pass',flush=True);del model;gc.collect()
    args.output.write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
