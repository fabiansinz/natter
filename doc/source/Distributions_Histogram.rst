Histogram Distribution
======================

The histogram distribution represents a histogram which has been
turned into a density. The parameters are the bin nodes and the
probability mass assigned to each bin. Data points will be assigned to
that bin node which is closest to it. Note that this means that bins
can have different size, and that the bin node may not lie in the
center of a bin. For the leftmost and rightmost bin, the bins are
chosen symmetrically around the bin node. Everything which is outside
the first bin node minus half its distance to the second is not
considered in binning. The same holds true for the rightmost bin. 



.. autoclass:: natter.Distributions.Histogram
   :members: sample,  pdf, ppf,  all, copy, parameters
