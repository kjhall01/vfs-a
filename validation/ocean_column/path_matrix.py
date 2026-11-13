"""Reproduce path-work controls against the unchanged finite-work solver.

Run from an external validation copy with an installed VFS wheel. Cases run
serially so wall times are interpretable; failed solves remain JSON results.
"""
import argparse,subprocess,sys,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--long',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cases=[('probe',[]),('finite300',['--method','finite']),('path150',['--dt','150']),('path75',['--dt','75']),('path600',['--dt','600']),('path32',['--layers','32']),('order5',['--order','5']),('cooling',['--scenario','cooling']),('rain',['--scenario','rain_heating']),('cautious',['--method','finite','--cautious'])]
    if a.long:cases.append(('long',['--hours','48']))
    for name,args in cases:
        subprocess.run([sys.executable,str(Path(__file__).with_name('path_campaign.py')),*args,'--output',str(a.output/(name+'.json'))],check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
if __name__=='__main__':main()
