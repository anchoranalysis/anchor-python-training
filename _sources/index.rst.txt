======================
anchor-python-training
======================

This is the documentation of **anchor-python-training**.


Introduction
============

Scripts in Python to train various types of CNN-models from images (using PyTorch).

They are installed collectively as a package `anchor_python_training`.

Usage
=====

Each `.py` script in the top-level directory of `src/anchor_python_training <https://github.com/anchoranalysis/anchor-python-training/tree/master/src/anchor_python_training>`_ is designed as a command-line application.

Please first install the package, by:

* `pip install .` (in the root of the checked out repository) or
* `pip install git+https://github.com/anchoranalysis/anchor-python-training.git`

A script can then be called from the command-line with the `-m` argument, ala::

   python -m anchor_python_training.script_top_level_name --somearg

Top-Level Scripts
=================

- :ref:`train_autoencoder <autoapi/train_autoencoder/index:Input Arguments>` - trains a CNN-based autoencoder from images (`source <https://github.com/anchoranalysis/anchor-python-training/blob/master/src/anchor_python_training/train_autoencoder.py>`_).


API
===

.. toctree::
   :maxdepth: 4


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
