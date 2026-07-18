# ParallelTrees
Open MPI implementation of **Parallel Trees**, a set of scalable, multi-NIC collective
algorithms for large-message MPI `Scatter`, `Gather`, and `Broadcast` on
GPU-accelerated clusters. This repository contains our patched Open MPI
source (adding Parallel Trees as selectable `coll_tuned` algorithms) and the
Python scripts used to generate latency figures from OSU Micro-Benchmarks
(OMB) output.


## Prerequisites

This repository only covers building the **patched Open MPI**. It assumes UCX
and OSU Micro-Benchmarks are already built separately. Tested with:

- UCX 1.19.0 (built with `--with-cuda`)
- CUDA 12.6.2
- GCC 12.3.1
- Ubuntu 24.04.4 LTS, kernel 6.8.0-124-generic
- `nvidia_peermem` kernel module loaded (for GPUDirect RDMA — recommended for
  all algorithms, not just Parallel Trees)

## Building Open MPI

Clone this repository and build the patched Open MPI against your existing
UCX and CUDA installations:

```bash
git clone https://github.com/AmirrezaBarati/ParallelTrees.git
cd ParallelTrees
tar -xzvf openmpi-4.1.8_Parallel_Trees.tar.gz
cd openmpi-4.1.8_Parallel_Trees/
mkdir build && cd build
../configure --prefix=<build_dir> \
  --with-ucx=<ucx_build_dir> --with-cuda=<cuda_path> \
  --enable-shared --with-verbs --with-hwloc=internal \
  --with-libevent=internal --with-pmix=internal \
  --enable-mca-no-build=btl-uct,btl-openib \
  --with-slurm --with-libfabric=no --with-ofi=no

make -j16 LDFLAGS+="-lcudart"
make install
```

Replace `<build_dir>`, `<ucx_build_dir>`, and `<cuda_path>` with your actual
install locations. Build time is typically under an hour on a modern
multi-core machine.

### Verifying the build

After installation, confirm the Parallel Trees algorithms are registered:

```bash
<build_dir>/bin/ompi_info --param coll tuned -l 9 | grep -A2 scatter_algorithm
```

You should see algorithm indices 4 and 5 available for Scatter (full-mesh and
NVSwitch variants, respectively) beyond Open MPI's stock algorithms 1–3.
Analogous new indices are added for Gather (4, 5) and Broadcast (11, 12),
alongside Open MPI's existing baseline algorithms for each collective.

### Running a collective with Parallel Trees

Once built and linked against OMB, a Scatter run using Parallel Trees on a
full-mesh topology (`g=4`, 4 GPUs/node) looks like:

```bash
mpirun -n 64 --map-by ppr:4:node \
  -x UCX_PROTO_ENABLE=y -x UCX_MAX_RMA_LANES=1 -x UCX_MAX_RNDV_RAILS=1 \
  --mca pml ucx --mca coll ^hcoll --mca btl ^openib \
  --mca pml_ucx_tls any --mca pml_ucx_devices any \
  --mca coll_tuned_use_dynamic_rules 1 \
  --mca coll_tuned_scatter_algorithm 4 \
  <omb_build>/libexec/osu-micro-benchmarks/get_local_rank \
  <omb_build>/libexec/osu-micro-benchmarks/mpi/collective/osu_scatter \
  -m 1048576:33554432 -x 50 -i 200 -f -d cuda
```

Swap `--mca coll_tuned_scatter_algorithm 4` for `5` to use the NVSwitch
variant, or for the algorithm number of any Open MPI baseline to reproduce a
comparison run. If your cluster requires explicit per-rank NIC pinning (e.g.,
one HCA per GPU), set `UCX_NET_DEVICES` accordingly before launching, either
per-rank via your job launcher or with a wrapper script of your own.

## Python plotting scripts

The scripts in `scripts/` turn raw OMB output logs into latency figures.
Each script expects OMB output files to already be collected for every
algorithm being compared, at the GPU count being plotted.

### Usage

```bash
python <script>.py {16,32,64} [OPTIONS]
```

**Positional argument:**

| Argument | Description |
|---|---|
| `{16,32,64}` | Total GPU count to plot (required). Must match a GPU count for which OMB logs were collected. |

**Options:**

| Flag | Description | Default |
|---|---|---|
| `-t`, `--chart-type {bar,line}` | Plot style | `bar` |
| `-s`, `--setup {fullmesh,nvs}` | Intra-node topology / platform | `fullmesh` |
| `-y`, `--yscale {log,linear}` | Y-axis scale | `log` |
| `-o`, `--output PATH` | Custom output PDF path | `[TODO: default naming scheme, e.g. <collective>_<gpu_count>_<setup>.pdf]` |
| `-h`, `--help` | Show help message | — |

### Examples

Plot Scatter latency at 64 GPUs on the full-mesh platform, log-scale bar chart:

```bash
python scripts/plot_scatter.py 64 -s fullmesh -y log -t bar
```

Plot Broadcast latency as a line chart on the NVSwitch platform at 32 GPUs,
saved to a custom path:

```bash
python scripts/plot_bcast.py 32 -s nvs -t line -o figures/bcast_32_nvs.pdf
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
