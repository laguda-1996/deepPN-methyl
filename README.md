# deepPN-methyl
We have developed a tool that can predict the methylation of m6A from the sequencing data of pacbio and nanopore, and achieved excellent performance in both training data and test data. For pacbio data, the methylation sequence detected by IPDsummary software was used for training; for nanopore data, the current information detected by Tombo software was used for training.

# Depetency
Python3.7 pysam>=0.15 biopython>=1.78 pandas>=0.24 numpy>=1.18 sklearn>=0.22 pytorch>=1.5  

# Usage
## 1. Pretreatment
### 1.1 Pacbio-Pretreatment



### 1.2 Nanopore-Pretreatment
  First use `tombo resdeepsignal extract quiggle path/to/fast5s/ genome.fasta --processes 4 --num-most-common-errors 5`.Please refer to https://github.com/nanoporetech/tombo ， this step processes the fast5 files down the machine, and the processing results are re written back to the fast5 files. 
  Then use `deepsignal extract`.Please refer to https://github.com/bioinfomaticsCSU/deepsignal, this step will extract the sequence information and current information from the fast5 file.
  
