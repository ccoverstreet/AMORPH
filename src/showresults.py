# Uses actual DNest4 stuff if it's installed - otherwise
# uses dnest_functions.py as a fallback
try:
    import dnest4.classic as dn4
except:
    import dnest_functions as dn4

dn4.postprocess()

import display
display.display()

