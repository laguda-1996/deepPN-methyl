# deepPN-methyl
We have developed a tool that can predict the methylation of m6A from the sequencing data of pacbio and nanopore, and achieved excellent performance in both training data and test data. For pacbio data, the methylation sequence detected by IPDsummary software was used for training; for nanopore data, the current information detected by Tombo software was used for training.

# Depetency
Python3.7 pysam>=0.15 biopython>=1.78 pandas>=0.24 numpy>=1.18 sklearn>=0.22 pytorch>=1.5  
