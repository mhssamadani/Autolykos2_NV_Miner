# Cuda Miner for Autolykos v2 (Ergo) for Nvidia GPUs

Cuda miner for [ergoplatform.com](https://github.com/ergoplatform)

You can find OpenCL miner at:  [OpenCL miner](https://github.com/mhssamadani/Autolykos2_AMD_Miner)

## Stratum Proxy

In order to use this miner with a stratum pool, a stratum proxy is needed.
- Install [Ergo Stratum Proxy](https://github.com/mhssamadani/ErgoStratumProxy)
- Update proxy's [`client.js`](https://github.com/mhssamadani/ErgoStratumProxy/blob/main/client.js)  file:
  - [port running port](https://github.com/mhssamadani/ErgoStratumProxy/blob/94b4561fbb857b3dbd227535bca75db311de8d66/client.js#L139) (optional. default is 3000)
  - [Pool address and port](https://github.com/mhssamadani/ErgoStratumProxy/blob/94b4561fbb857b3dbd227535bca75db311de8d66/client.js#L7)
- Start proxy
- In the miner's config file set node address to the proxy's address
 (by default this address is: ```{ "node" : "https://127.0.0.1:3000" }```)

## Prerequisites (Linux)
(For Ubuntu 16.04 or 18.04)

To compile you need the following:

1. CUDA Toolkit: see [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
2. CUDA Driver compatible with installed Toolkit: see [compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver)
3. libcurl library: to install run
```
$ apt install libcurl4-openssl-dev
```
4. OpenSSL 1.0.2 library: to install run
```
$ apt install libssl-dev
```

## Install (Linux)

1. Change directory to `Autolykos2_NV_Miner/secp256k1`
2. Run `make`

If `make` completed successfully there will appear an executable
`Autolykos2_NV_Miner/secp256k1/auto.out` and (if not already present)
a config file `Autolykos2_NV_Miner/secp256k1/config.json` with stub contents.

## Install (Windows 64-bit)

1. Install compatible pair of MS Visual Studio C++ toolchain and CUDA toolkit [compatibility table for latest CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
2. Build libcurl from sources with Visual Studio toolchain [instruction](https://medium.com/@chuy.max/compile-libcurl-on-windows-with-visual-studio-2017-x64-and-ssl-winssl-cff41ac7971d)
3. Download OpenSSL 1.0.2 [installer from slproweb.com](https://slproweb.com/download/Win64OpenSSL-1_0_2u.exe)
4. Edit `secp256k1/buildwin.cmd` file, change `OPENSSL_DIR`, `LIBCURL_DIR` to your libcurl and OpenSSL directories, change `CUDA_COMPUTE_ARCH` to GPU code architecture you want
5. Find `vcvars64.bat` script, it should be in `VISUAL_STUDIO_INSTALL_DIRECTORY\VC\Auxiliary\Build`
6. Run cmd.exe, run `vcvars64.bat` script, then change dir to secp256k1, then run `buildwin.cmd`
7. If everything went good, `miner.exe` should appear in `secp256k1` directory 
8. If `miner.exe` can't find `nvml.dll`, add `C:\Program Files\NVIDIA Corporation\NVSMI` to your PATH environment variable before running.


## Run (Linux)

- To run the miner you should pass a name of a configuration file `[YOUR_CONFIG]` as an optional argument
- If the filename is not specified, the miner will try to use `Autolykos2_NV_Miner/secp256k1/config.json` as a config
- The configuration file must contain json string of the following structure:  
`{ "node" : "https://127.0.0.1:9052" }`

To run the miner on all available CUDA devices type:
```
$ <YOUR_PATH>/Autolykos2_NV_Miner/secp256k1/auto.out [YOUR_CONFIG]
```

To choose CUDA devices change and use `runner.sh` or directly change environment variable `CUDA_VISIBLE_DEVICES`

## Run (Windows 64-bit)

- Create a config.json file in miner directory with following structure:
`{ "node" : "https://127.0.0.1:9052" }`

To change CUDA devices available to the miner change environment variable `CUDA_VISIBLE_DEVICES` , for example ` set CUDA_VISIBLE_DEVICES="0,1" `

## HTTP Info

Miner has a HTTP info page located at `http://miningnode:36207` (one can change default port by adding `-DHTTPAPI_PORT XXXX` to Makefile).

It outputs total hashrate, and per-GPU hashrates, power usages and temperatures in JSON format (relies on NVML, can fail if NVML fails - if so, JSON contains error field).

# Donations and Support

Note that the miner is free to use and we do not charge any fee from what you mine. To support all the work we're doing, we welcome donations from ERGO miners!

Bitcoin: 3KkwygpCLs1oEi9aTozFxYunoASV6ZrykJ

Bitcoin: bc1q7flay376e5mcp4ljjxpdp7r6p8yajcjm5mu6wd

ERGO: 9fFUw6DqRuyFCv13nQyoDuDz4TiR4GvVvWRcSvqzs39eBVcb5S1
