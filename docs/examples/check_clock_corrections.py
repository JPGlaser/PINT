# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check the state of PINT's clock corrections
#
# In order to do precision pulsar timing, it is necessary to know how the observatory clocks differ from a global time standard so that TOAs can be corrected. This requires PINT to have access to a record of measured differences. This record needs to be updated when new data is available. This notebook demonstrates how you can check the status of the clock corrections in your version of PINT. The version in the documentation also records the state of the PINT distribution at the moment the documentation was generated (which is when the code was last changed).

# %%
import pint.observatory

# %%
# hide annoying INFO messages
from loguru import logger as log
import sys
import pint.logging

logfilter = pint.logging.LogFilter()
log.remove()
log.add(sys.stderr, level="WARNING", filter=logfilter, colorize=True)


# %%
pint.observatory.list_last_correction_mjds()

# %%
pint.observatory.check_for_new_clock_files_in_tempo12_repos()

# %%
