using GridapML
using Documenter

DocMeta.setdocmeta!(GridapML, :DocTestSetup, :(using GridapML); recursive=true)

makedocs(;
    modules=[GridapML],
    authors="Santiago Badia <santiago.badia@gmail.com>",
    repo="https://github.com/BadiaLab/GridapML.jl/blob/{commit}{path}#{line}",
    sitename="GridapML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BadiaLab.github.io/GridapML.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/BadiaLab/GridapML.jl",
    devbranch="main",
)
