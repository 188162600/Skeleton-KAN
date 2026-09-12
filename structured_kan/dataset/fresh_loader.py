"""Load only verified fresh-seed artifacts, never the 30k-point audit pool."""
import json
from pathlib import Path
import numpy as np
import torch

from .fresh_sampling import VERSION,SEEDS,SPLITS,digest,array_hash,sampling_seed
from .regression import RegressionData,Split


def load_seed_data(row,folder,points=10000):
    folder=Path(folder)
    receipt=json.loads((folder/'fresh_receipts'/(row['audit_id'][:20]+'.json')).read_text())
    assert receipt['status']=='complete' and receipt['version']==VERSION
    assert receipt['case_id']==row['case_id'] and receipt['points_per_split']==points
    assert receipt['points_total']==30*points and receipt['sampling_sets']==30
    assert receipt['independent_fresh_draws'] and not receipt['reused_old_pool']
    data=[];fingerprints=[];ids=[];all_times=[]
    for seed in SEEDS:
        record=receipt['files'][str(seed)];path=folder/record['file']
        assert digest(path)==record['sha256'],path
        with np.load(path,allow_pickle=False) as arrays:
            splits=[]
            for split in SPLITS:
                x,y=arrays['x_'+split],arrays['y_'+split]
                assert x.dtype==y.dtype==np.float64 and x.shape==(points,row['variables']) and y.shape==(points,1)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                part=record['splits'][split]
                assert array_hash(x)==part['x_sha256'] and array_hash(y)==part['y_sha256']
                assert int(arrays['sampling_seed_'+split])==part['sampling_seed']==sampling_seed(row['case_id'],seed,split)
                fingerprints.append(part['x_sha256']);ids.append(arrays['sample_ids_'+split])
                if 'observation_times_'+split in arrays:all_times.append(arrays['observation_times_'+split])
                splits.append(Split(torch.from_numpy(x.copy()),torch.from_numpy(y.copy())))
            data.append(RegressionData(*splits,row['equation'],seed))
    assert len(set(fingerprints))==30
    assert np.array_equal(np.sort(np.concatenate(ids)),np.arange(30*points))
    if all_times:assert np.unique(np.concatenate(all_times)).size==30*points
    return data,receipt
