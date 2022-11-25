include("config_fem.jl")
include("../../../src/RB/RB.jl")
include("../../../src/RB/PoissonS.jl")

function config_RB(ϵₛ::Float)

  println("Setting up the RB solver, steady Poisson problem")
  println("Don't forget to modify config_fem!")

  training_percentage = 0.8
  considered_snaps = Int(floor(FEMInfo.nₛ*training_percentage))::Int
  online_RHS = false
  use_norm_X = false
  nₛ_MDEIM = min(20, considered_snaps)::Int
  import_offline_structures = true
  save_offline_structures = true
  save_results = true
  postprocess = false

  RBPaths = ROMPath(FEMInfo.Paths)
  RBInfo = ROMInfoS{1}(FEMInfo, RBPaths, considered_snaps, ϵₛ,
    use_norm_X, online_RHS, nₛ_MDEIM, postprocess,
    import_offline_structures, save_offline_structures, save_results)
  RBVars = ROMMethodS(RBInfo, Float)

  RBInfo, RBVars

end
