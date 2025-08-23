---
dateCreated: 2025-08-08
dateModified: 2025-08-09
---

This document describes NCCL's build system architecture, compilation process, and packaging mechanisms. It covers the Makefile orchestration, source compilation, and distribution packaging for various platforms.

## Overview

NCCL's build system is designed to compile the library from source and package it for distribution in various formats. The system follows a hierarchical structure with a root Makefile that coordinates builds across different subsystems.

Sources: [Makefile1-32]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32))

## Build System Architecture

![](Build%20System%20Architecture.png)

The build system is organized as a hierarchical structure with a root Makefile that delegates to subdirectories.

The build system is designed to:

1. Compile the NCCL library and associated components
2. Generate distribution packages (RPM, Debian, TXZ archives)
3. Handle license file distribution
4. Support different build configurations and targets

Sources: [Makefile1-32]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32))

## Root Makefile Orchestration

The root Makefile serves as the entry point for all build operations, delegating specific tasks to subdirectories.

### Key Targets

| Target       | Description                          | Implementation                  |
| ------------ | ------------------------------------ | ------------------------------- |
| `default`    | Default build target (builds source) | `src.build`                     |
| `install`    | Installs built components            | `src.install`                   |
| `clean`      | Cleans all build artifacts           | Calls clean on all subtargets   |
| `test.build` | Builds test components               | Depends on `src.build`          |
| `lic`        | Copies license files                 | Copies LICENSE.txt to build dir |

### Build Directory Configuration

The build directory is configurable through the `BUILDDIR` variable, which defaults to `./build`. The absolute path is computed and passed to subtargets.

```

BUILDDIR ?= $(abspath ./build)

ABSBUILDDIR := $(abspath $(BUILDDIR))

```

### Delegation Pattern

The root Makefile uses a pattern rule to delegate targets to subdirectories:

```

src.%:

${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

  

pkg.%:

${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

```

This allows invoking targets in subdirectories using the pattern `<subdir>.<target>`.

Sources: [Makefile1-32]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32))

## Source Build System

The source build system is responsible for compiling the NCCL library and associated components.

![](Source%20Build%20System.png)

The source build system handles:

1. Compilation of NCCL source files into shared and static libraries
2. Installation of header files and libraries
3. Configuration of compiler flags and architecture-specific optimizations
4. Dependency tracking for incremental builds

Sources: [Makefile24-25]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L24-L25](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L24-L25))

## Packaging System

The packaging system creates distribution packages for different platforms.

![](Packaging%20System.png)

### Package Types

The packaging system supports three main package formats:

1. **Debian packages (.deb)** - For Debian-based distributions
2. **RPM packages (.rpm)** - For Red Hat-based distributions
3. **TXZ archives (.txz)** - Generic tar+xz archives for manual installation

### Package Components

Each package format typically produces multiple package components:

| Package     | Contents                                        | Description                                 |
| ----------- | ----------------------------------------------- | ------------------------------------------- |
| [Runtime    | `libnccl.so](http://Runtime%7C%60libnccl.so).*` | Shared library for runtime use              |
| Development | Headers, symlinks                               | Development files for building against NCCL |
| Static      | `libnccl_static.a`                              | Static library for static linking           |

### License Handling

The packaging system ensures that license files are included in all packages:

```
pkg.debian.prep: lic
pkg.txz.prep: lic
```

Sources: [Makefile27-31]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L27-L31](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L27-L31))

## Build and Package Workflow

The following diagram illustrates the typical workflow for building and packaging NCCL:

![](Build%20and%20Package%20Workflow.png)

This workflow shows how the build system handles different user commands, from basic compilation to package generation.

Sources: [Makefile1-32]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L1-L32))

## Build Configuration Options

The build system supports various configuration options that can be passed as environment variables or make arguments:

| Option         | Description                 | Default                 |
| -------------- | --------------------------- | ----------------------- |
| `BUILDDIR`     | Build output directory      | `./build`               |
| `DEBUG`        | Enable debug build          | Not set                 |
| `CUDA_HOME`    | CUDA installation directory | Auto-detected           |
| `NVCC_GENCODE` | CUDA architecture targets   | Auto-detected           |
| `NVCC`         | CUDA compiler               | `$(CUDA_HOME)/bin/nvcc` |
| `CXXFLAGS`     | C++ compiler flags          | Platform-specific       |
| `PREFIX`       | Installation prefix         | `/usr/local`            |

These options can be used to customize the build process for different environments and requirements.

Sources: [Makefile10-11]([https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L10-L11](https://github.com/NVIDIA/nccl/blob/7c12c627/Makefile#L10-L11))

## Conclusion

NCCL's build system provides a flexible and modular approach to compiling and packaging the library. The hierarchical structure with delegated responsibilities allows for clean separation of concerns between source compilation and package generation.

The system supports multiple package formats for different distribution platforms, ensuring that NCCL can be easily deployed in various environments.
