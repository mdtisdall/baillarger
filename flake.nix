{
  description = "baillarger";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.gh
          ];
          # uv uses the devShell's Python and never downloads its own.
          env = {
            UV_PYTHON = "${python}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
          } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            # Binary wheels from PyPI (numpy, scipy, numba) link to libstdc++
            # and zlib, which NixOS and the Nix shell do not put on the
            # loader path.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
          };
        };
      });
}
