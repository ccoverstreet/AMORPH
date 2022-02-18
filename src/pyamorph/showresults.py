# Uses actual DNest4 stuff if it's installed - otherwise
# uses dnest_functions.py as a fallback
try:
    import dnest4.classic as dn4
except:
    from . import dnest_functions as dn4


def amorph_postprocess():
    dn4.postprocess()

    from . import display
    display.display()

if __name__ == "__main__":
    amorph_postprocess()

