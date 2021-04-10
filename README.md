# deepPN-methyl
We have developed a tool that can predict the methylation of m6A from the sequencing data of pacbio and nanopore, and achieved excellent performance in both training data and test data. For pacbio data, the methylation sequence detected by IPDsummary software was used for training; for nanopore data, the current information detected by Tombo software was used for training.

# Depetency
Python3.7 pysam>=0.15 biopython>=1.78 pandas>=0.24 numpy>=1.18 sklearn>=0.22 pytorch>=1.5  

Note：the current prediction and training scripts must be CUDA

The pre training model is in the network disk, please download it by yourself
链接：https://pan.baidu.com/s/1hRof76lXcGYzX7JPU_UEAA 
提取码：1234 


# Usage
## 1. Pretreatment
### 1.1 Pacbio-Pretreatment
bam files from the lower machine and the reference genome were first aligned using pbmm2 alignment software.

`pbmm2 index [options] <ref.fa|xml> <out.mmi>`

`pbmm2 align [options] <ref.fa|xml|mmi> <in.bam|xml|fa|fq> [out.aligned.bam|xml]`


Please refer to https://github.com/PacificBiosciences/pbmm2 ，
The sequences of interest are then called from the alignment file using the kinetics tool


`ipdSummary aligned.bam --reference ref.fasta --identify m6A,m4C --gff basemods.gff`

(please refer to https://github.com/PacificBiosciences/kineticsTools) and the result is written to a csv file from which the desired sequences are extracted written to a fasta file.
If a site is known and needs to be verified as a methylated site, extract a sequence with a total length of 41 BP each 20bp up - and downstream of the site, write to a fasta file.

### 1.2 Nanopore-Pretreatment
  First use `tombo resquiggle path/to/fast5s/ genome.fasta --processes 4 --num-most-common-errors 5`
  
  Please refer to https://github.com/nanoporetech/tombo ， this step processes the fast5 files down the machine, and the processing results are re written back to the fast5 files. 
  Then use `deepsignal extract --fast5_dir fast5s.al/ --reference_path GCF_000146045.2_R64_genomic.fna --write_path fast5s.al.CpG.signal_features.17bases.rawsignals_360.tsv --corrected_group RawGenomeCorrected_001 --nproc 10
`

Please refer to https://github.com/bioinfomaticsCSU/deepsignal, this step will extract the sequence information and current information from the fast5 file.
  
## 2. Predict
Use our trained model to predict directly and write the score value of the result into the file which is named by yourself.  
```
python predict.py predict_fa (.fa) -model_path (checkpoint) -outfile (file_name)
```
## 3. Train by yourself
Use our pretreatment process to train a new model with your known A site.We also provide a negative set for your training.More hyperparametric adjustment instructions can be obtained using the `python train.py -h`  
```
python train.py -pos_fa (pos.fa) -neg_fa (neg.fa) -outdir (dir_name)
```
### Additional
If you want to get not only the predicted score, but also the various evaluation indicators of the prediction, you can use `python test.py`.Please refer to `python test.py -h`
