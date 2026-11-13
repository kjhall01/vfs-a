"""Reproducible short and twelve-hour research controls, serial by default."""
import argparse,subprocess,sys,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cases=[('legacy', ['--solver','legacy']),('refresh',['--solver','refresh']),('budget',[]),('dt150',['--dt','150']),('dt75',['--dt','75']),('dt600',['--dt','600']),('layers32',['--layers','32']),('cooling',['--scenario','cooling']),('rain',['--scenario','rain_heating']),('order5',['--order','5']),('long',['--hours','12'])]
 for name,args in cases:
  subprocess.run([sys.executable,str(Path(__file__).with_name('consistent_campaign.py')),*args,'--output',str(a.output/(name+'.json'))],check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
if __name__=='__main__':main()
