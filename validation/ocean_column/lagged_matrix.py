"""Semi-implicit qualification matrix; serial reproduction with installed VFS."""
import argparse,subprocess,sys,os
from pathlib import Path
CASES=[('dt300',[]),('dt150',['--dt','150']),('dt75',['--dt','75']),('dt18',['--dt','18.75']),('sub4',['--substeps','4']),('sub16',['--substeps','16']),('factor05',['--reduction','.5']),('long',['--substeps','4','--hours','12']),('cooling',['--substeps','4','--scenario','cooling']),('rain',['--substeps','4','--scenario','rain_heating']),('layers32',['--substeps','4','--layers','32']),('reference18',['--reference','--dt','18.75']),('reference300',['--reference']),('factor05_fine',['--dt','18.75','--reduction','.5']),('column_control',['--funding','column']),('initial_local',['--funding','local','--predictors','0','--reduction','.5'])]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for name,args in CASES:
  subprocess.run([sys.executable,str(Path(__file__).with_name('lagged_campaign.py')),*args,'--output',str(a.output/(name+'.json'))],check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
if __name__=='__main__':main()
