import sys
import io
import platform
import numpy as np
import astropy
from pint.models import get_model_and_toas
import pint.fitter
import pint.utils
import pint.logging

par = """# Created: 2023-05-23T10:08:47.546618
# PINT_version: 0.9.5
# User: M. A. Krishnakumar (kkma)
# Host: poe
# OS: Linux-5.4.0-148-generic-x86_64-with-glibc2.29
# Format: pint
PSRJ                           J0621+1002
EPHEM                               DE440
CLK                          TT(BIPM2019)
UNITS                                 TDB
START              56568.2121563848386458
FINISH             59421.4003363883475926
TIMEEPH                              FB90
T2CMETHOD                        IAU2000B
DILATEFREQ                              N
DMDATA                                  N
NTOA                                   10
CHI2                                  0.0
RAJ                      6:21:22.01017540 0 0.02108535635446286832
DECJ                    10:02:29.39636000 0 1.28655830483269539855
PMRA                   154.99933932091736 0 38.32827139688301
PMDEC                   853.9002881758452 0 161.0480808201861
PX                                    0.0
POSEPOCH           54999.9998161703821150
F0                  34.657407161944268063 0 5.8220569308e-10
F1              -6.8024291921798985053e-17 0 2.3431820241319046872e-18
PEPOCH             54999.9998161703821150
CORRECT_TROPOSPHERE                         N
PLANET_SHAPIRO                          N
NE_SW                                 7.9
SWM                                   0.0
DM                  36.567498299041078198 0 0.00099999995348440769
DMEPOCH            55000.0000000000000000
DMX                                   6.5
DMX_0001             -0.02286374033884119 1 0.004209721846073168
DMXR1_0001         56568.1621563848311806
DMXR2_0001         56568.2621565318113426
DMX_0002             0.004038234230037191 1 0.003956496936912034
DMXR1_0002         59421.3503361464245254
DMXR2_0002         59421.4503363883559375
BINARY                                 DD
PB                   8.319084026957771904 0 2.7466991161234e-07
PBDOT                                 0.0
A1                     12.032352334870641 0 0.00022070836169778854
A1DOT                                 0.0
ECC                  0.002384353926582889 0 3.976882262223037e-05
EDOT                                  0.0
T0                 55145.5433555870737050 0 0.08005422103296240617
OM                  182.54567292992324803 0 0.00396216643901861707
OMDOT              0.77923849834938746777 0 0.44091740076385298584
A0                                    0.0
B0                                    0.0
GAMMA                                 0.0
DR                                    0.0
DTH                                   0.0
TZRMJD             58092.0258119028393056
TZRSITE                             lofar
TZRFRQ                 167.28515859378945
"""

tim = """FORMAT 1
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2013-10-03-04:57.add_clean_calibP_tscr.ar 125.097656 56568.212156384518097 921.223  lofar  -fe unknown -be LOFAR -f unknown_LOFAR -bw 14.06 -tobs 899.81 -tmplt newtemplates/J0621+1002.ar -gof 1.07 -nbin 1024 -nch 72 -prof_snr 7.26 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2013-10-03-04:57.add_clean_calibP_tscr.ar 139.200325 56568.212156531479532 922.440  lofar  -fe unknown -be LOFAR -f unknown_LOFAR -bw 14.06 -tobs 899.81 -tmplt newtemplates/J0621+1002.ar -gof 1.1 -nbin 1024 -nch 72 -prof_snr 10.82 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2013-10-03-04:57.add_clean_calibP_tscr.ar 153.237640 56568.212156394256012 739.059  lofar  -fe unknown -be LOFAR -f unknown_LOFAR -bw 14.06 -tobs 899.81 -tmplt newtemplates/J0621+1002.ar -gof 1.07 -nbin 1024 -nch 72 -prof_snr 11.45 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2013-10-03-04:57.add_clean_calibP_tscr.ar 167.250770 56568.212156415574501 440.814  lofar  -fe unknown -be LOFAR -f unknown_LOFAR -bw 14.06 -tobs 899.81 -tmplt newtemplates/J0621+1002.ar -gof 1.05 -nbin 1024 -nch 72 -prof_snr 9.46 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2013-10-03-04:57.add_clean_calibP_tscr.ar 181.187528 56568.212156466085537 1236.402  lofar  -fe unknown -be LOFAR -f unknown_LOFAR -bw 14.06 -tobs 899.81 -tmplt newtemplates/J0621+1002.ar -gof 1.19 -nbin 1024 -nch 72 -prof_snr 5.98 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2021-07-26-09:28.add_clean_calibP_tscr.ar 125.066372 59421.400336245320750 602.268  lofar  -fe unknown -be COBALT -f unknown_COBALT -bw 14.06 -tobs 899.56 -tmplt newtemplates/J0621+1002.ar -gof 1.07 -nbin 1024 -nch 72 -prof_snr 9.46 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2021-07-26-09:28.add_clean_calibP_tscr.ar 139.279255 59421.400336197322133 416.165  lofar  -fe unknown -be COBALT -f unknown_COBALT -bw 14.06 -tobs 899.56 -tmplt newtemplates/J0621+1002.ar -gof 0.938 -nbin 1024 -nch 72 -prof_snr 11.99 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2021-07-26-09:28.add_clean_calibP_tscr.ar 153.226188 59421.400336153624831 275.830  lofar  -fe unknown -be COBALT -f unknown_COBALT -bw 14.06 -tobs 899.56 -tmplt newtemplates/J0621+1002.ar -gof 0.9 -nbin 1024 -nch 72 -prof_snr 18.62 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2021-07-26-09:28.add_clean_calibP_tscr.ar 167.261939 59421.400336146117558 266.152  lofar  -fe unknown -be COBALT -f unknown_COBALT -bw 14.06 -tobs 899.56 -tmplt newtemplates/J0621+1002.ar -gof 1.06 -nbin 1024 -nch 72 -prof_snr 15.27 -sys LOFAR.150
/data/kkma/postprocessing/data//LOFAR/J0621+1002/J0621+1002.2021-07-26-09:28.add_clean_calibP_tscr.ar 181.458939 59421.400336388027380 587.558  lofar  -fe unknown -be COBALT -f unknown_COBALT -bw 14.06 -tobs 899.56 -tmplt newtemplates/J0621+1002.ar -gof 0.981 -nbin 1024 -nch 72 -prof_snr 9.88 -sys LOFAR.150
"""


pint.logging.setup("WARNING")
print(f"# python_version: {sys.version}")
print(f"# numpy_version: {np.__version__}")
print(f"# astropy_version: {astropy.__version__}")
print(pint.utils.info_string(prefix_string="# "))


#Computer network name
print(f"Computer network name: {platform.node()}")
#Machine type
print(f"Machine type: {platform.machine()}")
#Processor type
print(f"Processor type: {platform.processor()}")
#Platform type
print(f"Platform type: {platform.platform()}")
#Operating system
print(f"Operating system: {platform.system()}")
#Operating system release
print(f"Operating system release: {platform.release()}")
#Operating system version
print(f"Operating system version: {platform.version()}")

# dmx values that are gotten by D Kaplan and Krishnakumar
dmx_dlk = np.array([-0.022862539247389357, 0.0040382142830872585])
dmx_kk = np.array([-0.02286136775682614, 0.0040378097214163])


# m, t = get_model_and_toas("J0621+1002_kk_DMX_tdb.par", "J0621+1002_two-testToAs.tim")
m, t = get_model_and_toas(io.StringIO(par), io.StringIO(tim))
print(f"Free parameters: {m.free_params}")
f = pint.fitter.Fitter.auto(t, m)
f.fit_toas()
dmxoutput = pint.utils.dmxparse(f)
for i in range(len(dmxoutput["bins"])):
    print(
        f"{dmxoutput['bins'][i]}: ({dmxoutput['r1s'][i]}-{dmxoutput['r2s'][i]}): {dmxoutput['dmxs'][i]+dmxoutput['mean_dmx']}: {dmxoutput['dmx_verrs'][i]}"
    )
print(
    f"Differences wrt DLK: {(dmxoutput['dmxs'].value+dmxoutput['mean_dmx'].value-dmx_dlk)/dmx_dlk}"
)
print(
    f"Differences wrt KK: {(dmxoutput['dmxs'].value+dmxoutput['mean_dmx'].value-dmx_kk)/dmx_kk}"
)
