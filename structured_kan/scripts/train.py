"""End-to-end integration example, not a replacement research dataset."""
import argparse
import json
import math
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',help='Filename under data/train_config, or an explicit path')
    parser.add_argument('--catalogue',type=Path,default=Path(__file__).resolve().parents[1]/'configs/example_catalogue.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--seed-start',type=int,default=421)
    parser.add_argument('--points',type=int,default=10_000)
    parser.add_argument('--outer-steps',type=int,default=50)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--compile-mode',choices=['off','default','reduce-overhead'],default='off')
    parser.add_argument('--initialization-noise',type=float,default=.001)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.config:
        candidate=Path(args.config)
        if not candidate.is_file():candidate=Path(__file__).resolve().parents[1]/'data/train_config'/args.config
        if json.loads(candidate.read_text()).get('schema') in ('structured-kan.mlp-domain80.v1','structured-kan.fourier-mfn-domain80.v1','structured-kan.multkan-domain80.v1','structured-kan.smpf-structure-domain80.v1'):
            from .neural_benchmark import inspect_config,run_queue
            if args.validate_only:
                path,config,rows=inspect_config(args.config)
                print(json.dumps(dict(config=str(path),method=config['method'],equations=len(rows),valid=True)));return
            run_queue(args.config);return
        if json.loads(candidate.read_text()).get('schema')=='structured-kan.domain300-training.v1':
            from .benchmark import inspect_config,run_queue
            if args.validate_only:
                path,config,rows=inspect_config(args.config)
                print(json.dumps(dict(config=str(path),method=config['method'],equations=len(rows),valid=True)));return
            run_queue(args.config);return
        if __package__:
            from .configuration import inspect_config
        else:
            from configuration import inspect_config
        inspected=inspect_config(args.config)
        if args.validate_only:
            print(json.dumps(inspected['summary'],indent=2));return
        # Importing snapshots must not change archived route sharing, silently
        # substitute the example model, or reuse audit samples as fresh seeds.
        raise RuntimeError(inspected['config']['execution']['reason']+
            ' Configuration references are valid, but the archived backend is not yet migrated.')
    if args.output is None:
        parser.error('--output is required for the integration demo')
    import torch
    from ..dataset import sample_box, regression_metrics, geometric_mean_nmse
    from ..model import StructuredKANBuilder
    from ..optimizer import fit_seed_batch
    from ..structure_builder import Catalogue
    if args.seeds<1 or args.initialization_noise<0:parser.error('invalid seeds or initialization noise')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable; CPU requires explicit --device cpu')
    catalogue=Catalogue.load(args.catalogue)
    seeds=list(range(args.seed_start,args.seed_start+args.seeds))
    raw=[sample_box(lambda x:torch.cos(x[:,0])+x[:,0]*x[:,1],
                    [[-math.pi,math.pi],[-1,1]],equation='cos_plus_product_demo',seed=s,points=args.points) for s in seeds]
    standardized=[data.standardize() for data in raw]
    data=[item[0] for item in standardized]
    transforms=[item[1] for item in standardized]
    results=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for entry in catalogue.entries:
        models=[]
        for index,seed in enumerate(seeds):
            model=StructuredKANBuilder(2,device=device).build(entry['structure'],train_inputs=data[index].train.x)
            generator=torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(args.initialization_noise*torch.randn(parameter.shape,generator=generator,device=device,dtype=parameter.dtype))
            models.append(model)
        if args.validate_only:
            for model,values in zip(models,data):
                prediction=model(values.train.x.to(device))
                assert prediction.shape==values.train.y.shape and bool(torch.isfinite(prediction).all())
            results.append(dict(builder=entry['id'],parameters=models[0].parameter_count(),status='validated'))
            continue
        fit=fit_seed_batch(models,data,outer_steps=args.outer_steps,
                           compile_mode=None if args.compile_mode=='off' else args.compile_mode)
        records=[]
        for index,(seed,model) in enumerate(zip(seeds,models)):
            transform=transforms[index]
            metrics={}
            with torch.no_grad():
                for split in ('train','validation','test'):
                    normalized=getattr(data[index],split)
                    prediction=model(normalized.x.to(device))*transform['y_std'].to(device)+transform['y_mean'].to(device)
                    metrics[split]=regression_metrics(prediction,getattr(raw[index],split).y.to(device))
            record=dict(seed=seed,parameters=model.parameter_count(),metrics=metrics,
                        inner_iterations=fit.inner_iterations[index],function_evaluations=fit.function_evaluations[index])
            records.append(record)
            # Numeric names avoid interpreting catalogue labels as filesystem paths.
            filename=f'builder_{len(results):03d}_seed_{seed}.pt'
            torch.save(dict(structure=entry['structure'],state_dict={n:t.detach().cpu() for n,t in model.state_dict().items()},
                            normalization=transform,result=record),args.output/filename)
        validation_score=sum(math.log(max(r['metrics']['validation']['nmse'],1e-300)) for r in records)/len(records)
        results.append(dict(builder=entry['id'],seeds=records,mean_log_validation_nmse=validation_score,
                            test_gnmse=geometric_mean_nmse([r['metrics']['test']['nmse'] for r in records])))
    report=dict(status='validated' if args.validate_only else 'completed',dataset='integration_demo_not_benchmark',
        catalogue_sha256=catalogue.sha256,seeds=seeds,points_per_split=args.points,outer_steps=args.outer_steps,
        normalization='train-only input/output standardization',results=results)
    if not args.validate_only:
        selected=min(results,key=lambda r:r['mean_log_validation_nmse'])
        report['selected_builder']=selected['builder'];report['selected_test_gnmse']=selected['test_gnmse']
    (args.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
