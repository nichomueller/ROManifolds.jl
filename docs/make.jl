using Documenter
using Literate
using ROM

makedocs(;
    modules=[ROM],
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Usage" => "usage.md",
        "Steady" => "steady.md",
        "Transient" => "transient.md",
        "Contributing" => "contributing.md",
        "Reference" => ["public_api.md","types.md","functions.md"],
    ],
    sitename="ROM.jl",
)

deploydocs(
  repo = "git@github.com:nichomueller/ROM.jl.git",
)
