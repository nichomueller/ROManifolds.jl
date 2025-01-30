using Documenter
using ROM

fem_interface = [
  "utils.md",
  "dof_maps.md",
  "tproduct.md",
  "param_data_structures.md",
  "param_algebra.md",
  "param_fe_spaces.md",
  "param_steady.md",
  "param_odes.md",
  ]

rom_interface = [
  "rbsteady.md",
  "rbtransient.md",
]

makedocs(;
    modules=[ROM],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "FEM Interface" => fem_interface,
        "ROM Interface" => rom_interface,
        "Contributing" => "contributing.md",
    ],
    sitename="ROM.jl",
)

# deploydocs(
#   repo = "github.com:nichomueller/ROM.jl.git",
# )
