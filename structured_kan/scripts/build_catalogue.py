"""Remote construction entry point; never starts neural training."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['ks_ies','v2','ted','fgw_sum','acuos2'],required=True,
                   help='ks_ies is KS-IES; v2 is a compatibility alias for frozen records')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--heldout-domain',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--k',type=int,default=18)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--threads',type=int,default=10)
    p.add_argument('--pair-timeout-seconds',type=float,default=30.)
    a=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]=str(a.threads)
    os.environ['NATIVE_PREPROCESS_CACHE']=str(a.output.parent/'preprocessing.sqlite')
    from ..structure_builder.build import build_catalogue
    try:
        method='v2' if a.method=='ks_ies' else a.method
        result=build_catalogue(a.source,a.output,method=method,k=a.k,heldout_domain=a.heldout_domain,
                               workers=a.workers,pair_timeout_seconds=a.pair_timeout_seconds)
    except Exception as error:
        from ..structure_builder.io import write
        # Do not overwrite a completed run when the caller reuses its path.
        if not isinstance(error,FileExistsError) and a.output.is_dir():
            write(a.output/'failure.json',dict(error=repr(error)))
        raise
    print(json.dumps(dict(method=a.method,heldout_domain=a.heldout_domain,builders=result['builder_count'],
                         output=str(a.output/'catalogue.json'),construction=result['construction'])))


if __name__=='__main__':main()
