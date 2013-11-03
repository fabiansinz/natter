The Data class
==============

The Data class stores data vectors in a member variable *X*. Each
column of *X* is a data sample. Each Data object has a history where
previous processing steps are protocoled. Data objects support slicing
and subindexing just like numpy arrays. Additionally, data objects
can be exponentiated.

Data overloads the plus operator: If two Data objects are added, they
are joined.

.. moduleauthor:: Fabian Sinz <fabee@bethgelab.org>

.. autoclass:: natter.DataModule.Data
   :members:

