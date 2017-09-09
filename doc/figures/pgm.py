"""
PGM for the general problem using daft (http://daft-pgm.org/)
"""

import matplotlib
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import daft

pgm = daft.PGM([9., 5.], origin=[-2.5, -2.5])

# Create the nodes
pgm.add_node(daft.Node('alpha', r'$\boldsymbol{\alpha}_{\rm amorph}$', 0., 2.))
pgm.add_node(daft.Node('x', r'$\boldsymbol{\theta}_i^{\rm amorph}$', 0., 0.))
pgm.add_node(daft.Node('data', r'$\boldsymbol{y}$', 2., 0., observed=True))
pgm.add_node(daft.Node('N', r'$N_{\rm amorph}$', -2., 0.))
pgm.add_node(daft.Node('hidden', r'', -1., 0., scale=0, plot_params={'alpha':0}))

# Right hand side
pgm.add_node(daft.Node('x2', r'$\boldsymbol{\theta}_i^{\rm xtal}$', 4., 0.))
pgm.add_node(daft.Node('alpha2', r'$\boldsymbol{\alpha}_{\rm xtal}$', 4., 2.))
pgm.add_node(daft.Node('N2', r'$N_{\rm xtal}$', 6., 0.))
pgm.add_node(daft.Node('hidden2', r'', 5., 0., scale=0, plot_params={'alpha':0}))

# Top and bottom
pgm.add_node(daft.Node('sigma', r'$\boldsymbol{\sigma}$', 2., 2.))
pgm.add_node(daft.Node('sie', r'$\boldsymbol{\theta}_{\rm bg}$', 2., -2.))

# Add the edges
pgm.add_edge('alpha', 'x')
pgm.add_edge('x', 'data')
pgm.add_edge('N', 'hidden')

pgm.add_edge('alpha2', 'x2')
pgm.add_edge('x2', 'data')
pgm.add_edge('N2', 'hidden2')

pgm.add_edge('sigma', 'data')
pgm.add_edge('sie', 'data')

# Add the plates
pgm.add_plate(daft.Plate([-1., -1., 2., 2.], label=r'Amorphous gaussians \newline $i=1, ..., N_{\rm amorph}$'))

# Add the plates
pgm.add_plate(daft.Plate([3., -1., 2., 2.], label=r'Crystalline gaussians\newline $i=1, ..., N_{\rm xtal}$'))

pgm.render()
pgm.figure.savefig("pgm.pdf")
pgm.figure.savefig("pgm.svg")


