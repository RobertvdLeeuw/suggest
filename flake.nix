{
  description = "Suggest: music recommender - uv2nix-based dev environment.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    { self, nixpkgs, uv2nix, pyproject-nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;

      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python312;

      # uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = ./.;
        config.deps = {
          default = true;
          db = true;
          collecter = true;
          suggester = true;
          frontend = true;
        };
      };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      # Fixups uv2nix can't infer on its own - see
      # https://pyproject-nix.github.io/uv2nix/FAQ.html
      # Pattern is the same for each: the package's own build backend isn't
      # declared as a build dependency, so add it via resolveBuildSystem.
      pyprojectOverrides = final: prev: {
        numba = prev.numba.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb_2022_0 ];
        });

        jukemirlib = prev.jukemirlib.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });
        fire = prev.fire.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });
        jukebox = prev.jukebox.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });
        wget = prev.wget.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });
        psycopg2 = prev.psycopg2.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });
        jaconv = prev.jaconv.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; };
        });

        hatchling = prev.hatchling.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem { setuptools = [ ]; wheel = [ ]; };
        });

        hatch-vcs = prev.hatch-vcs.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem {
              hatchling = [ ];
              hatch-vcs = [ ];
              setuptools = [ ]; # Fallback
            };
        });

        spotdl-lean = prev.spotdl-lean.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ])
            ++ final.resolveBuildSystem {
              poetry = [ ];
              setuptools = [ ];
              wheel = [ ];
            };
        });
      };

      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
        inherit python;
      }).overrideScope (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        overlay
        pyprojectOverrides
      ]);

      rocmLibPath = lib.makeLibraryPath [
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocm-runtime
        pkgs.rocmPackages.rocm-device-libs
        pkgs.rocmPackages.hip-common
        pkgs.rocmPackages.hipblas
        pkgs.rocmPackages.hipfft
        pkgs.rocmPackages.hipsolver
        pkgs.rocmPackages.hipsparse
        pkgs.rocmPackages.rocblas
        pkgs.rocmPackages.miopen
        pkgs.rocmPackages.rccl
        pkgs.stdenv.cc.cc.lib
        pkgs.libsndfile
      ];

      commonShell = {
        packages = [
          python
          pkgs.uv

          pkgs.spotdl
          pkgs.libsndfile

          pkgs.postgresql_16
          pkgs.postgresql_16.pg_config
          pkgs.postgresql16Packages.pgvector

          pkgs.rocmPackages.rocm-smi
          pkgs.rocmPackages.rocm-runtime
          pkgs.rocmPackages.clr
        ];

        env = {
          UV_PYTHON_DOWNLOADS = "never"; # Prevent uv from managing Python downloads.
          ROCM_PATH = "${pkgs.rocmPackages.clr}";
          HIP_PATH = "${pkgs.rocmPackages.clr}";
          LD_LIBRARY_PATH = "${rocmLibPath}";
        };

        # Only expose packages declared in this shell, no global PYTHONPATH leakage.
        shellHook = "unset PYTHONPATH";
      };
    in
    {
      packages.${system}.default =
        pythonSet.mkVirtualEnv "suggest-env" workspace.deps.default;

      devShells.${system} = {
        # Direct pip access, nixpkgs' Python instead of the uv2nix venv.
        # For anything uv2nix can't build cleanly yet - ROCM/tensorflow spikes etc.
        impure = pkgs.mkShell {
          inherit (commonShell) packages shellHook;
          env = commonShell.env // {
            UV_PYTHON = python.interpreter;
          } // lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH =
              "${rocmLibPath}:${lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1}";
          };
        };

        # Editable install of the workspace itself, built purely through Nix -
        # local file changes apply without a rebuild.
        default =
          let
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              root = "$REPO_ROOT";
            };

            editablePythonSet = pythonSet.overrideScope (
              lib.composeManyExtensions [
                editableOverlay
                (final: prev: {
                  suggest = prev.suggest.overrideAttrs (old: {
                    # Only these paths need to trigger a rebuild.
                    src = lib.fileset.toSource {
                      root = old.src;
                      fileset = lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/src")
                        (old.src + "/tests")
                      ];
                    };

                    # Hatchling needs `editables` explicitly declared for editable
                    # builds under Nix (implicit under normal pip/PEP 660 flows).
                    nativeBuildInputs = old.nativeBuildInputs
                      ++ final.resolveBuildSystem { editables = [ ]; };
                  });
                })
              ]
            );

            virtualenv =
              editablePythonSet.mkVirtualEnv "suggest-dev-env" workspace.deps.all;
          in
          pkgs.mkShell {
            inherit (commonShell) packages;
            env = commonShell.env // {
              UV_NO_SYNC = "1"; # Venv is already built above.
              UV_PYTHON = "${virtualenv}/bin/python";
              LD_LIBRARY_PATH = "${rocmLibPath}";
            };

            shellHook = commonShell.shellHook + ''
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              export PATH="${virtualenv}/bin:$PATH"
              export PYTHONPATH="$REPO_ROOT"
            '';
          };
      };
    };
}
