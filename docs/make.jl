using Documenter
using ROM

makedocs(;
    modules=[ROM],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "FEM Interface" => "fem_interface.md",
        "ROM Interface" => "rom_interface.md",
        "Contributing" => "contributing.md",
    ],
    sitename="ROM.jl",
)

# deploydocs(
#   repo = "github.com:nichomueller/ROM.jl.git",
# )
