using Documenter
using ROM

makedocs(;
    modules=[ROM],
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Home" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "ROM.Utils" => "Utils.md",
        "ROM.DofMaps" => "DofMaps.md",
        "ROM.TProduct" => "TProduct.md",
        "ROM.ParamDataStructures" => "ParamDataStructures.md",
        "ROM.ParamAlgebra" => "ParamAlgebra.md",
        "ROM.ParamFESpaces" => "ParamFESpaces.md",
        "ROM.ParamSteady" => "ParamSteady.md",
        "ROM.ParamODEs" => "ParamODEs.md",
        "ROM.RBSteady" => "RBSteady.md",
        "ROM.RBTransient" => "RBTransient.md",
    ],
    sitename="ROM.jl",
    doctest = false,
    warnonly = [:cross_references,:missing_docs],
    checkdocs = :exports,
)

deploydocs(
  repo = "git@github.com:nichomueller/ROM.jl.git",
)
