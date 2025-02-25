using Documenter
using ROManifolds

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
    modules=[ROManifolds],
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Home" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "FEM Interface" => fem_interface,
        "ROManifolds Interface" => rom_interface,
        "Contributing" => "contributing.md",
    ],
    sitename="ROManifolds.jl",
    warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo = "github.com:nichomueller/ROManifolds.jl.git",
)
