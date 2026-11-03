# shell.nix

{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  }
}:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    uv

    # Python tooling / LSPs
    ruff
    basedpyright

    git
    gcc
    pkg-config

    cudaPackages.cuda_nvcc
  ];

   shellHook = ''
     # export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
     # export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
     export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH


     export TRITON_PTXAS_PATH=${pkgs.cudaPackages.cuda_nvcc}/bin/ptxas


     # Triton finds libcuda via /sbin/ldconfig, which NixOS doesn't have
     if [ -d /run/opengl-driver/lib ]; then
       export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
     fi
   '';
}

