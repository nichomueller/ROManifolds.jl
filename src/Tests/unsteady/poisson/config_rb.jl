include("config_fem.jl")
include("../../../src/RB/RB.jl")
include("../../../src/RB/PoissonST.jl")

function config_RB(ϵₛ::Float, ϵₜ::Float)

  println("Setting up the RB solver, steady Poisson problem")
  println("Don't forget to modify config_fem!")

  training_percentage = 0.8
  considered_snaps = Int(floor(FEMInfo.nₛ*training_percentage))::Int
  online_RHS = false
  use_norm_X = false
  nₛ_MDEIM = min(20, considered_snaps)::Int
  t_red_technique = "ST-HOSVD"
  import_offline_structures = false
  save_offline_structures = true
  save_results = true
  postprocess = false
  st_MDEIM = true
  functional_MDEIM = false
  adaptivity = false

  RBPaths = ROMPath(FEMInfo.Paths)
  RBInfo = ROMInfoST{1}(FEMInfo, RBPaths, considered_snaps, ϵₛ, ϵₜ,
    use_norm_X, t_red_technique, online_RHS, nₛ_MDEIM, postprocess,
    import_offline_structures, save_offline_structures, save_results,
    st_MDEIM, functional_MDEIM, adaptivity)
  RBVars = ROMMethodST(RBInfo, Float)

  RBInfo, RBVars

end
