Examples from natter paper
==============================


This part of the documentation contains all examples used in [SinzEtAl2012]_.
The code of all examples is also located in the Examples directory.

.. contents::

Loading data
------------
In the data loading example we show how to load data from a simple (gzipped) ascii file
and how to use the given sampling iterators and methods to create Data objects.

from ``Examples/loadData.py``:

.. literalinclude:: ../../Examples/loadData.py

Preprocessing
-------------
In the preprocessing example we show how to prepare data for modelling. The mean
value is subtracted, the DC-component projected out and the remaining AC components
are whitened.

from ``Examples/preprocessing.py``:

.. literalinclude:: ../../Examples/preprocessing.py

Training ICA model
------------------
In the first training example we fit a product of exponential power distributions
to the data. This is what ICA (independent component analysis) is doing.

from ``Examples/trainingICA.py``:

.. literalinclude:: ../../Examples/trainingICA.py

Training Lp-sperically symmetric model
--------------------------------------
In the second training example we fit an Lp-spherical symmetric distribution to
the data.

from ``Examples/trainingLp.py``:

.. literalinclude:: ../../Examples/trainingLp.py

Training complete linear model
------------------------------
In this example we fit a complete linear model, consisting of a linear transformation
(FastICA filter in this case) and a distribution.

from ``Examples/trainingCompleteLinearModel.py``:

.. literalinclude:: ../../Examples/trainingCompleteLinearModel.py

Radial factorization
--------------------
Here we give a simple example how to use radial factorization, a nonlinear transform.

from ``Examples/radialFactorization.py``:

.. literalinclude:: ../../Examples/radialFactorization.py

Testing a model
---------------
This is a simple example on how to fit a model using a training set and then evaluate
the quality of the fit by comparing the average log-loss of the training and test
data set under the optimized model.

from ``Examples/testing.py``:

.. literalinclude:: ../../Examples/testing.py

.. [SinzEtAl2012] Fabian Sinz, Joern-Philipp Lies, Sebastian Gerwinn, and 
   Matthias Bethge, *NATTER: A Python Natural Image Statistics Toolbox*, *in preparation*

