#!/usr/bin/env python
# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
# Actually it's not disabled? Unclear what the above is supposed to mean.
import os
import shutil
import sys
from io import StringIO
from pathlib import Path

import pytest

from pint.scripts import event_optimize
from pinttestdata import datadir

@pytest.mark.skip(reason="Doesn't actually test much of anything and is slow")
def test_result(tmp_path):
    parfile = os.path.join(datadir, "PSRJ0030+0451_psrcat.par")
    eventfile_orig = os.path.join(
        datadir, "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
    )
    temfile = os.path.join(datadir, "templateJ0030.3gauss")
    eventfile = tmp_path / "event.fits"
    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)

    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
    try:
        os.chdir(tmp_path)
        cmd = "{0} {1} {2} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10".format(
            eventfile, parfile, temfile
        )
        # FIXME: tries to create a file next to the tim file
        event_optimize.main(cmd.split())
        lines = sys.stdout.getvalue()
        # Need to add some check here.
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout
