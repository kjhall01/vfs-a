"""Record CPU, numeric backends and installed distributions without secrets."""
import argparse
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sysconfig

import numpy
import scipy
import torch
import vfs


def environment():
    command = ['lscpu'] if platform.system() == 'Linux' else ['sysctl', '-n', 'machdep.cpu.brand_string']
    cpu = subprocess.run(command, capture_output=True, text=True)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        numpy.show_config()
        scipy.show_config()
    return dict(
        platform=platform.platform(), python=platform.python_version(), cpu=cpu.stdout.strip(),
        cpu_query_error=cpu.stderr.strip() if cpu.returncode else None,
        torch=str(torch.__version__), torch_config=torch.__config__.show(),
        torch_parallel_info=torch.__config__.parallel_info(),
        cpu_capability=torch.backends.cpu.get_cpu_capability(), numeric_libraries=config.getvalue(),
        source=str(Path(vfs.__file__).resolve()),
        environment={k: os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY', 'MKL_CBWR',
                     'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        distributions={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--checkout', type=Path)
    args = parser.parse_args()
    source = Path(vfs.__file__).resolve()
    assert source.is_relative_to(Path(sysconfig.get_path('purelib')).resolve()), source
    if args.checkout:
        assert not source.is_relative_to(args.checkout.resolve()), source
    result = environment()
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
