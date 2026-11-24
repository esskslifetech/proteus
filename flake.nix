{
  description = "Proteus - Human Genome Simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # Rust toolchain
        rust-toolchain = pkgs.rust-bin.stable.latest.default;

        # Development shell
        devShell = pkgs.mkShell {
          name = "proteus-dev";
          buildInputs = with pkgs; [
            rust-toolchain
            python311
            python311Packages.maturin
            python311Packages.pip
            python311Packages.virtualenv
            cmake
            eigen
            sundials
            highs
            pkg-config
            openssl
            git
            cargo-nextest
            cargo-audit
            cargo-deny
          ];

          # Python packages for development
          nativeBuildInputs = with pkgs.python311Packages; [
            numpy
            scipy
            pandas
            polars
            pyarrow
            torch
            transformers
            biopython
            pytest
            hypothesis
            ruff
            mypy
          ];

          RUST_BACKTRACE = 1;
          CARGO_TERM_COLOR = "always";
          PYTHONPATH = "/proteus/python:${PYTHONPATH}";
        };

        # Build the Rust workspace
        rust-build = pkgs.stdenv.mkDerivation {
          name = "proteus-rust-build";
          src = self;
          nativeBuildInputs = [ rust-toolchain cargo-nextest ];
          buildPhase = ''
            cargo build --workspace --release
          '';
          installPhase = ''
            mkdir -p $out/lib
            cp target/release/libproteus*.so $out/lib/ 2>/dev/null || true
            cp target/release/libproteus*.dylib $out/lib/ 2>/dev/null || true
            cp target/release/libproteus*.dll $out/lib/ 2>/dev/null || true
          '';
        };

        # Build C++ components
        cpp-build = pkgs.stdenv.mkDerivation {
          name = "proteus-cpp-build";
          src = self;
          nativeBuildInputs = [ cmake ];
          buildInputs = [ eigen sundials highs pybind11 ];
          buildPhase = ''
            cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$out
            cmake --build cpp/build -j${pkgs.lib.makeJobs}
            cmake --install cpp/build
          '';
        };

        # Python package
        python-package = pkgs.python311Packages.buildPythonPackage {
          name = "proteus-python";
          src = self;
          format = "pyproject.toml";
          buildInputs = with pkgs.python311Packages; [
            numpy
            scipy
            pandas
            polars
            pyarrow
            torch
            transformers
            biopython
            pysam
            pyfaidx
            gffutils
            intervaltree
          ];
          propagatedBuildInputs = with pkgs.python311Packages; [
            fastapi
            uvicorn
            typer
            rich
            jinja2
            weasyprint
            pylatex
            rocrate
            pydantic
            jsonschema
            protobuf
          ];
          doCheck = false;  # Tests run separately
        };

      in
      {
        devShells.default = devShell;
        devShells.rust = devShell.overrideAttrs (old: {
          name = "proteus-rust-dev";
        });
        devShells.cpp = devShell.overrideAttrs (old: {
          name = "proteus-cpp-dev";
          buildInputs = old.buildInputs ++ [ pkgs.gdb pkgs.lldb ];
        });

        packages.default = pkgs.buildEnv {
          name = "proteus";
          paths = [ rust-build cpp-build python-package ];
        };

        # CI checks
        checks = {
          rust-fmt = pkgs.runCommand "rust-fmt" { buildInputs = [ rust-toolchain ]; } ''
            cargo fmt --all -- --check
          '';
          rust-clippy = pkgs.runCommand "rust-clippy" { buildInputs = [ rust-toolchain ]; } ''
            cargo clippy --workspace --all-targets --all-features -- -D warnings
          '';
          rust-test = pkgs.runCommand "rust-test" { buildInputs = [ rust-toolchain cargo-nextest ]; } ''
            cargo nextest run --workspace
          '';
          cpp-build-check = pkgs.runCommand "cpp-build" { buildInputs = [ cmake eigen sundials highs ]; } ''
            cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
            cmake --build cpp/build -j${pkgs.lib.makeJobs}
          '';
          python-test = pkgs.runCommand "python-test" { 
            buildInputs = [ python-package pkgs.python311Packages.pytest pkgs.python311Packages.hypothesis ];
          } ''
            cd python && pytest -v
          '';
        };
      }
    );
}