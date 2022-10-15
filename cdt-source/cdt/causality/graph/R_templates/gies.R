# MIT License
#
# Copyright (c) 2018 Diviyan Kalainathan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

library(methods)
library(pcalg)
set.seed(as.integer('{SEED}'))

dataset <- read.csv(file='{FOLDER}{FILE}', header=FALSE, sep=",");
loaded_targets <- read.csv(file='{FOLDER}{FILE_FLAT_TARGETS}', header=FALSE, sep=",");
loaded_target_lengths <- read.csv(file='{FOLDER}{FILE_FLAT_TARGET_LENGTHS}', header=FALSE, sep=",");
loaded_target_indices <- read.csv(file='{FOLDER}{FILE_TARGET_INDICES}', header=FALSE, sep=",");

if({SKELETON}){
  fixedGaps <- read.csv(file='{FOLDER}{GAPS}', sep=",", header=FALSE) # NULL
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}else{
  fixedGaps = NULL
}

# convert flattened targets into correct data type
targets <- list()
ctr <- 1
for(i in 1:nrow(loaded_target_lengths)){
	l <- loaded_target_lengths[i, 1];
	tar <- as.vector(as.double(loaded_targets[ctr:(ctr + l - 1), 1]));
	ctr <- ctr + l;
	if(l==1){
		if(tar[[1]]==c(-1)){
			# None (observational data) represented as integer(0)
			tar <- integer(0);
		}
	}
	targets[[i]] <- tar;
}

# convert target indices into correct data type
target_indices <- as.vector(as.double(loaded_target_indices[,1]))

# run algo
score <- new("{SCORE}", data=dataset, targets=targets, target.index=target_indices)
result <- pcalg::gies(score, labels=score$getNodes(), targets=score$getTargets(), fixedGaps=fixedGaps)
gesmat <- as(result$essgraph, "matrix")
gesmat[gesmat] <- 1
  #gesmat[!gesmat] <- 0
write.csv(gesmat, row.names=FALSE, file = '{FOLDER}{OUTPUT}');
