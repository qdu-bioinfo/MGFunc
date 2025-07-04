# Microbiome GPU-based Function inference

# Contents

- [Introduction](#introduction)
- [System requirement and dependency](#system-requirement-and-dependency)
- [Installation guide](#installation-guide)
- [Usage](#usage)
- [Example dataset](#example-dataset)
- [Reference databases](#reference-databases)
- [Recommended RAM size for memory-aware loading](#recommended-RAM-size-for-memory-aware-loading)
- [Contact](#contact)

# Introduction

Microbiome GPU-based Function inference (MGFunc) is a fast, scalable GPU-accelerated tool for functional prediction of microbiomes based on 16S rRNA gene sequencing. Detailed input format specifications can be found in the [Usage](#usage) section. It is optimized for ultra-large datasets, capable of handling insufficient RAM situations and automatically adapting to different GPU memory sizes.

# System requirement and dependency

## Hardware requirements

MGFunc requires NVIDIA GPU(s) or AMD GPU(s) in a x86 system with sufficient RAM. We recommend a computer with the following specs:

- **RAM: 16+ GB** 

The recommended RAM size varies with the iteration size parameter `-b`, which controls the number of samples processed per iteration. Please refer to [Recommended RAM size for memory-aware loading](#recommended-ram-size-for-memory-aware-loading) for details.

- **CPU: 8+ cores**

- **GPU: 1+ NVIDIA or AMD GPU**

The required GPU memory varies with the selected reference database, but MGFunc is compatible with the memory sizes of all commonly used GPUs. For details, see [Reference databases](#reference-databases).

## Software requirements

Operating system: Linux. MAC is not supported yet.

Basic compiler: C/C++ with OpenMP library. Most Linux releases have OpenMP already been installed in the system.

GPU complier: NVIDIA CUDA installation was referred to in the next section. HIP installation can be found [here](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).

# Installation guide

## CUDA Download and Install

a. Before installation, please check the current NVIDIA driver version (using the `nvidia-smi` command to see the maximum CUDA Toolkit version supported by the driver) and the Linux server version to ensure they support the required CUDA Toolkit version.

b. Visit the CUDA official website (<https://developer.nvidia.com/cuda-toolkit-archive>) to download the CUDA Toolkit version that matches your system environment.

E.g. The following uses **CUDA Version 12.2** and **g++ 8.3.1** as an example (** **Please choose the appropriate download link based on your system** **) 

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

**Note:** If multiple CUDA versions are installed on the system, you can switch between them using the following commands ( taking CUDA Toolkit 12.0 as an example ):

```
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-12.0 /usr/local/cuda
```

## CUDA version MGFunc Download and Install

### 1. Download the package

```
wget https://github.com/qdu-bioinfo/MGFunc/releases/download/v1.0/MGFunc.tar.gz
```

### 2. Install CUDA version

#### Automatic installation (recommended)

MGFunc provides a fully automatic installer for easy installation.

a. Extract the package:

	tar -zxvf MGFunc.tar.gz

b. Install

	cd MGFunc
	source install.sh


##### Tips for the Automatic installation

1. Please **“cd MGFunc”** before run the automatic installer.
2. The automatic installer only configures the environment variables to the default configuration files of “\~/.bashrc” or “\~/.zshrc”. If you want to configure the environment variables to other configuration file please use the manual installation.
3. If the automatic installer failed, MGFunc can still be installed manually by the following steps.

#### Manual installation

If the automatic installer failed, MGFunc can still be installed manually.

a. Extract the package:

	tar -zxvf MGFunc.tar.gz

b. Configure the environment variables (default environment variable configuration file is located at “\~/.bashrc” or “\~/.zshrc”)

	export MGFunc=Path to MGFunc
	export PATH="$PATH:$MGFunc/bin"

c. Then, active the environment variables

	source ~/.bashrc

d. Compile the source code:

	cd MGFunc
	make

### 3. Install HIP version (for AMD GPUs)

#### Automatic installation (recommended)

MGFunc provides a fully automatic installer for easy installation.

a. Extract the package:

	tar -zxvf MGFunc.tar.gz

b. Install

	cd MGFunc
	source install.sh hip


##### Tips for the Automatic installation

1. Please **“cd MGFunc”** before run the automatic installer.
2. The automatic installer only configures the environment variables to the default configuration files of “\~/.bashrc” or “\~/.zshrc”. If you want to configure the environment variables to other configuration file please use the manual installation.
3. If the automatic installer failed, MGFunc can still be installed manually by the following steps.

#### Manual installation

If the automatic installer failed, MGFunc can still be installed manually.

a. Extract the package:

	tar -zxvf MGFunc.tar.gz

b. Configure the environment variables (default environment variable configuration file is located at “\~/.bashrc” or “\~/.zshrc”)

	export MGFunc=Path to MGFunc
	export PATH="$PATH:$MGFunc/bin"

c. Then, active the environment variables

	source ~/.bashrc

d. Compile the source code:

	cd MGFunc
	make hip

# Usage

## **Input data format**

MGFunc requires microbiome abundance table (e.g. OTU table) to calculate the distances among microbiomes. Currently MGFunc supports OTUs of Greengenes, Greengenes2, SILVA, RefSeq and RDP. More reference database will be released soon. The input example is as follows:

```
sampleID	OTU_1	OTU_2	OTU_3	...	OTU_M
Sample_1	1	0	6	...	8
Sample_2	2	3	0	...	5
...	...	...	...	...	...
Sample_N	0	2	9	...	4
```

This input can be generated by Parallel-Meta Suite (PMS). URL for PMS is <https://github.com/qdu-bioinfo/parallel-meta-suite/tree/main>

## Using NVIDIA GPU for Computation

In this version, MGFunc assumes that **all GPUs on a single server have identical specifications** and **will utilize all GPU resources on single node**. Therefore, before running the program, please ensure that no other critical tasks are being executed on the server node to avoid disrupting other operations. Command of using NVIDIA GPU for computation is as follows:

```
MGFunc version: 1.0
Usage:
	MGFunc [Option] Value
Options: 

[Input options]
	-T    Input OTU count table (*.OTU.Count)
	-b    Number of iteration size, default: no iteration
	-D    Reference database, options: default is G (GreenGenes-13-8 (16S rRNA, 97% level)), or S (SILVA (16S rRNA, 97% level)),  or R (GreenGenes-2 (16S rRNA)), or D (RDP (16S rRNA)), or Q (Refseq (16S rRNA, 100% level))

[GPU options]
	-G    Use GPUs: 'a' for all available GPUs, or number of GPUs (default is 1)

[Data processing options]
	-N    Skip OTU count normalization: T for true, F for false (default is F)
	-S    Max NSTI value, OTUs above this value will be excluded (default is 2.0)

[Output options]
	-o    Output path (default is ./result)

[Other options]
	-t    Number of CPU threads, default: auto
	-h    Help
```

E.g. Calculate the functions of the **taxa.OTU.Count** file in the **/home** directory and output the result to "result" using Greengenes database.

### Single GPU Mode

Use `single GPU` to run on a single GPU.

```
MGFunc -T /home/taxa.OTU.Count -o result -D G
```

To skip copy number normalization, add `-N`:

```
MGFunc -T /home/taxa.OTU.Count -o result -D G -N T
```

To set a maximum NSTI threshold (e.g. 2.0), add `-S`:

```
MGFunc -T /home/taxa.OTU.Count -o result -D G -S 2.0
```

### Multi-GPU Mode

Use `multiple GPUs` to enable multi-GPU acceleration. By default, MGFunc will use all available GPUs.
To specify the number of GPUs to use `-G`.

Use all GPUs (default):

```
MGFunc -T /home/taxa.OTU.Count -o result -D G -G a
```

Use 4 GPUs:

```
MGFunc -T /home/taxa.OTU.Count -o result -D G -G 4
```

### Using memory-aware loading for Computation

MGFunc provides a **memory-aware loading mode** by adding the `-b` parameter. The iteration size can be specified with the `-b`.

E.g. Calculate the functions of the **taxa.OTU.Count** file in the **/home** directory and output the result to "result" using Greengenes database.

```
MGFunc -T /home/taxa.OTU.Count -o result -D G -b 400000
```

## Using AMD GPU for Computation

Usage of the HIP version is the same as the CUDA version. Simply replace `MGFunc` with `MGFunc_hip` in your command.

```
MGFunc_hip -T /home/taxa.OTU.Count -o result -D G
```

# Example dataset

Here, we provide a demo dataset in the "example" folder with 10,000 microbiomes. In this package, "taxa.OTU.Count" is the OTU count table. To run the demo, you can type the following command:

```
cd example
MGFunc -T taxa.OTU.Count -o output -D G
```

## Output description

We also provide the example output in example folder (reference database is Greengenes):

result.KO.Count : Rows are samples, columns are KO IDs. Values are KO copy counts per sample.

result.KO.Abd : Relative abundance table. Same structure as `result.KO.Count`，values are relative abundances per sample.

result.NSTI : Two-column table: sample name and NSTI value per sample.

# Reference databases

The table below lists the GPU memory requirements for the databases ([**Table 1**](#table-1-reference-databases)). For simplicity, we recommend that your GPU memory be at least 70% larger than the size of the selected database. However, this is not mandatory, as MGFunc is designed to automatically adapt to the memory capacities of all commonly used GPUs.

### Table 1. Reference databases

| ***\*Reference database\**** | ***# of microbes*** | ***# of  functions (**KEGG Ortholog**)*** | ***Size*** |
| ---------------------------- | ------------------- | ----------------------------------------- | ---------- |
| *Greengenes*                 | *99,322*            | *13,839*                                  | *5.12GB*   |
| *Greengenes2*                | *331,269*           | *13,839*                                  | *17.08GB*  |
| *Silva*                      | *152,265*           | *13,839*                                  | *7.85GB*   |
| *Refseq*                     | *101,192*           | *13,839*                                  | *5.22GB*   |
| *RDP*                        | *24,624*            | *13,839*                                  | *1.27GB*   |

# Recommended RAM size for memory-aware loading


Additionally, we provide a table of iteration sample sizes and recommended RAM sizes to guide performance optimization across a wide range of sample scales ([**Table 2**](#table-2-recommended-ram-size-for-memory-aware-loading)). The iteration size is controlled by the `-b` parameter; if `-b` is not specified, memory-aware loading is disabled by default.

### Table 2. Recommended RAM size for memory-aware loading

| ***\*RAM size\**** | ***# of iteration size*** |
| ------------------ | :-------------|
| *16GB*    | *10,000*     |
| *32GB*    | *50,000*     |
| *64GB*    | *100,000*    |
| *128GB*   | *200,000*    |
| *256GB*   | *400,000*    |
| *384GB*   | *600,000*    |
| *512GB*   | *800,000*    |
| *640GB*   | *1,000,000*  |

# Contact

Any problem please contact MGFunc development team 

```
Dr. Su,Xiaoquan   E-mail: suxq@qdu.edu.cn
```

