The Data class
==============

The Data class stores data vectors in a member variable *X*. Each
column of *X* is a data sample. Each Data object has a history where
previous processing steps are protocoled. Data objects support slicing
and subindexing just like numpy arrays. Additionally, data objects
can be exponentiated.  

.. autoclass:: natter.DataModule.Data
   :members: setHistory, norm, normalize, plot, addToHistory, scale, scaleCopy, mean, plotPatches, var, center, makeWhiteningVolumeConserving, cov, dnormdx, size, copy, save, append, numex, dim

