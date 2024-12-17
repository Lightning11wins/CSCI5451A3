# Assignment #3: K-Medoids Clustering Algorithm (CUDA)
## Algorithm Parallelization
After the setup, where we read the file and select the first k points as medoids, the algorithm consists of two essential phases.
In phase 1, we add each point to the cluster centered on the closest medoid. To parallelize this, we use the job of assigning a single point to a cluster. We divide points evenly between each thread of each block, using a 1D hirarchy since the threads are a 1D array.
In phase 2, we calculate the new medoid of each cluster by minimizing cluster size. To do this, we calculate the size of each point's cluster if that point were the new medoid. Then we use the CPU pick the new medoids. This is probably a bottleneck, but I wasn't able to find an algorithm that could effectively avoid this. Again, we divide points evenly between each thread of each block, using a 1D hirarchy since the threads are a 1D array.

## Timing
When clustering `small_cpd`, relative speeds are shown below:

### Time in Seconds per Program vs. Number of Clusters
|           | 1024     | 512      | 256      |
|-----------|----------|----------|----------|
| km_kuda   | 16.4117s | 31.4777s | 43.4433s |
| km_openmp | 4.5607s  | 4.5218s  | 7.1676s  |

### Analysis
After spending actual days trying to figure out why my program was so much slower, I've finally run out of time. The deadline is in a few minutes, and I still can't figure out what's wrong, so here are my best guesses.
- Guess 1: There's way more people using the GPUs right now than the CPUs. This could make my code run slower because it has to wait for time on the GPU. This theory is supported by the fact that the machine I'm using for this assignment periodically freezes, maybe due to being under intense load.
- Guess 2: My alorithm probably has a significant bug. This is the more likely reason, but at this point, I've spent so many hours looking for this bug that my mental health is starting to degrade, and I'm mentally and physically exausted.

### Conclusion
I've really given this assignment everything I could. I tried as hard as I could to get this to work, and if that's not good enough, then I guess I've failed... and I suppose I'm ok with that. I did my best.