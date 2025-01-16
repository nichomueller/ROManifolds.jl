using Documenter
using ROM

makedocs(;
    modules=[ROM],
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "Reference" => ["public_api.md","types.md","functions.md"],
    ],
    sitename="ROM.jl",
    checkdocs=:exports
)

deploydocs(
  repo = "git@github.com:nichomueller/ROM.jl.git",
  push_preview = true
)
