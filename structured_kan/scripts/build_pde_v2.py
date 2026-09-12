"""One shared Source-300 v2 construction and source-only private-map allocation."""
import json
import sys
import time
import traceback
from .benchmark import PROJECT,save,sha

def main():
    started=time.time();root=PROJECT.parent
    source=PROJECT/'data/equation_set/domain300/selected/domain300_with_splits.json'
    base=PROJECT/'data/results/pde_v2_source300_construction'
    destination=PROJECT/'data/catalogues/pde10/v2_source300_k18_dynamic_g.json'
    try:
        from ..structure_builder.build import build_catalogue
        from ..structure_builder.shared_affine_complexity import allocate
        group=build_catalogue(source,base,method='v2',k=18,heldout_domain='pde',workers=8)
        assert group['training_equations']==300 and group['builder_count']==18
        result=allocate(base/'catalogue.json',source);save(destination,result)
        save(root/'construction_receipt.json',dict(status='complete',seconds=time.time()-started,
            source_sha256=sha(source),base_sha256=sha(base/'catalogue.json'),
            allocated_sha256=sha(destination),source_rows=300,builders=18,heldout_rows_read=0))
    except BaseException as e:
        save(root/'construction_receipt.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
