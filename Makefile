INPUT = /export/scratch/CSCI-5451/assignment-3/small_gaussian.txt
NUM_CLUSTERS = 256
NUM_BLOCKS = 1
NUM_THREADS_PER_CLUSTER = 1
PARAMS = $(INPUT) $(NUM_CLUSTERS) $(NUM_BLOCKS) $(NUM_THREADS_PER_CLUSTER)

# cp /export/scratch/CSCI-5451/assignment-3/small_gaussian.txt .
# cp /export/scratch/CSCI-5451/assignment-3/small_cpd.txt .

CC = nvcc
SRC = cluster.cu
DEP = cluster.h
ZIP = fulle637.tar.gz
ZIP_DIR = fulle637
EXE = km_cuda

SUPRESS = -Xcudafe="--diag_suppress=177"

.PHONY: run all clean load $(EXE)

run: $(EXE)
	./$(EXE) $(PARAMS)

$(EXE): $(SRC) $(DEP)
	$(CC) -O3 -Xcompiler="-Wall" $(SUPRESS) $(SRC) -o $(EXE)

load:
	module load soft/cuda/local
	module initadd soft/cuda/local
	module rm soft/cuda
	module initrm soft/cuda

clean:
	rm -f $(EXE) $(ZIP) medoids.txt clusters.txt
	git repack -a -d --depth=2500 --window=2500
	git gc --aggressive --prune=now
	du -a | sort -n

submission: clean
	mkdir $(ZIP_DIR)
	cp . $(ZIP_DIR)
	tar --exclude='.gitignore' --exclude='hello_cuda.cu' -czvf $(ZIP) $(ZIP_DIR)

diff:
	diff clusters.txt clusters_correct.txt > clusters_diff.txt
	diff medoids.txt medoids_correct.txt > medoids_diff.txt
