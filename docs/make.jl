using Mabla
using Documenter

DocMeta.setdocmeta!(Mabla, :DocTestSetup, :(using Mabla); recursive=true)

makedocs(;
    modules=[Mabla],
    authors="Santiago Badia <santiago.badia@gmail.com>",
    repo="https://github.com/BadiaLab/Mabla.jl/blob/{commit}{path}#{line}",
    sitename="Mabla.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BadiaLab.github.io/Mabla.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/BadiaLab/Mabla.jl",
    devbranch="main",
)
