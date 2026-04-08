# Full Search Block Matching (FSBM)

In a block matching approach (BM), frames are divided into blocks. Each block of the current frame is matched to find the best matching block in a region of the preceding frames by minimizing the Sum of Absolute Differences (SAD). The simplest method is to use a Full Search algorithm (FSA), which finds the motion vector accurately by exhaustively calculating the SAD value for all elements of the search region.
