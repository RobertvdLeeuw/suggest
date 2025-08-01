{
  description = "Hello world flake using uv2nix";

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
    {
    self,
    nixpkgs,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    ...
    }:
    let
      inherit (nixpkgs) lib;

      # Load a uv workspace from a workspace root.
      # Uv2nix treats all uv projects as workspace projects.
      workspace = uv2nix.lib.workspace.loadWorkspace {
        workspaceRoot = ./.;
        config.deps = {
          default = true;
          db = true;
          collecter = true;
          suggester = true;
          frontend = true;
        };
      };  # Loading in pyproject and project data.

      # Create package overlay from workspace.
      overlay = workspace.mkPyprojectOverlay { 
        sourcePreference = "wheel";  # Wheel is best, apparently.
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups, uv can only do so much on its own.
      # - https://pyproject-nix.github.io/uv2nix/FAQ.html
      pyprojectOverrides = _final: _prev: { 
        # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
        # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
        numba = _prev.numba.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [ pkgs.tbb_2022_0 ];
        });

        jukemirlib = _prev.jukemirlib.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });
        fire = _prev.fire.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });
        jukebox = _prev.jukebox.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });
        wget = _prev.wget.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });
        psycopg2 = _prev.psycopg2.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });
        jaconv = _prev.jaconv.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ 
            _final.resolveBuildSystem { setuptools = []; };
        });

        hatchling = _prev.hatchling.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++
            _final.resolveBuildSystem { 
              setuptools = []; 
              wheel = [];
            };
        });

        hatch-vcs = _prev.hatch-vcs.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++
            _final.resolveBuildSystem { 
              hatchling = [];
              hatch-vcs = [];  # Often needed with hatchling
              setuptools = [];  # Fallback
            };
        });

        spotdl-lean = _prev.spotdl-lean.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++
            _final.resolveBuildSystem {
              poetry = [];  
              setuptools = [];
              wheel = [];
            };
          buildInputs = (old.buildInputs or []) ++ [
            # Add any system dependencies if needed
          ];
        });

      };

      # This example is only using x86_64-linux
      pkgs = nixpkgs.legacyPackages.x86_64-linux;  # Legacy just means we can use entire Nixpkgs.

      python = pkgs.python312;

      # Construct package set
      pythonSet =  # Further 'translation' for python package use. TODO: Research more later.
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
        (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
            pyprojectOverrides
          ]
        );


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

    in
      {
      # Nix build
      packages.x86_64-linux.default = pythonSet.mkVirtualEnv "venv" workspace.deps.default;

      apps.x86_64-linux = {  # Nix run
        default = {
          type = "app";
          program = "${self.packages.x86_64-linux.default}/bin/myapp";  # TODO: Sync with pyproj and other name than myapp
        };
      };

      devShells.x86_64-linux =  # Nix develop
        let 
          setup = {  # General/shared
            packages = [
              python
              pkgs.uv

              pkgs.spotdl
              pkgs.libsndfile

              pkgs.postgresql_16
              
              pkgs.rocmPackages.rocm-smi
              pkgs.rocmPackages.rocm-runtime
              pkgs.rocmPackages.clr
            ];

            env = {
              UV_PYTHON_DOWNLOADS = "never";  # Prevent uv from managing Python downloads
              
              ROCM_PATH = "${pkgs.rocmPackages.clr}";
              HIP_PATH = "${pkgs.rocmPackages.clr}";
              LD_LIBRARY_PATH = "${rocmLibPath}";
            };

            shellHook = ''
              unset PYTHONPATH  # Only expose py packages stated in here, no global imports.
              '';
          };
        in {
          # For stuff like figuring out tensorflow rocm, or a general playground. Direct pip access.

          # TODO: Make this inherit from default, or would that be bad abstraction? 
          # Or 'let generalSetup = {} in default, impure'?
          impure = pkgs.mkShell { 
            packages = setup.packages;
            env = setup.env // {
              UV_PYTHON = python.interpreter;  # Force uv to use nixpkgs Python interpreter
            } // lib.optionalAttrs pkgs.stdenv.isLinux {
                # Exposes C (.so) stuff for packages like numpy. dlopen(3)
                LD_LIBRARY_PATH = "${rocmLibPath}:${lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1}";
              };
            shellHook = setup.shellHook;
          };

          # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
          # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
          #
          # This means that any changes done to your local files do not require a rebuild.
          #
          # Note: Editable package support is still unstable and subject to change.
          default =
            let
              # Create an overlay enabling editable mode for all local dependencies.
              editableOverlay = workspace.mkEditablePyprojectOverlay { 
                # Use environment variable
                root = "$REPO_ROOT";
                # Optional: Only enable editable for these packages
                # members = [ "hello-world" ];
              };

              # Override previous set with our overrideable overlay.
              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  # Apply fixups for building an editable package of your workspace packages
                  (final: prev: {
                    hello-world = prev.hello-world.overrideAttrs (old: {
                      # It's a good idea to filter the sources going into an editable build
                      # so the editable package doesn't have to be rebuilt on every change.
                      src = lib.fileset.toSource {
                        root = old.src;
                        fileset = lib.fileset.unions [ 
                          (old.src + "/pyproject.toml")
                          (old.src + "/src")
                          (old.src + "/tests") 
                        ];
                      };

                      # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                      #
                      # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                      # This behaviour is documented in PEP-660.
                      #
                      # With Nix the dependency needs to be explicitly declared.
                      nativeBuildInputs =  # For build dependencies, but not runtime ones.
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });
                  })
                ]
              );

              # Build virtual environment, with local packages being editable.
              #
              # Enable all optional dependencies for development.
              virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;
            in
              pkgs.mkShell {
                packages = setup.packages;
                env = setup.env // {
                  UV_NO_SYNC = "1";  # We already have created a virtualenv.
                  UV_PYTHON = "${virtualenv}/bin/python";  # Force uv to use Python interpreter from venv
                  LD_LIBRARY_PATH = "${rocmLibPath}";
                };

                # TODO: This somehow also execs on impure shells. Figure out how/why and whether I would want a true split.
                shellHook = setup.shellHook + '' 
                  # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                  export REPO_ROOT=$(git rev-parse --show-toplevel)  # TODO: This could be useful for nvim or other tools to find project root.
                  
                  export PATH="${virtualenv}/bin:$PATH" # Let Python know about the virtualenv for imports
                  export PYTHONPATH="$REPO_ROOT"
                '';
              };
        };
    };
}
