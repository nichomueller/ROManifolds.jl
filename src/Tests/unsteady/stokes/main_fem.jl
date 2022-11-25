include("config_fem.jl")

function get_FEM_solution(FEMInfo::FOMInfo{ID}) where ID

  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{FEMInfo.D,FEMInfo.D}
  @assert FEMInfo.D == num_cell_dims(model) "Wrong dimension: change selected model"
  FEMSpace = get_FEMSpace(FEMInfo, model)::FOM{ID,FEMInfo.D}
  @inline FEMμ(μ::Vector) = FEM_solver(FEMInfo, get_operator(FEMInfo, model, μ)...)

  FEMSpace, FEMμ

end

function get_FEM_structures(
  FEMSpace::FOM{ID,D},
  FEMInfo::FOMInfo{ID},
  μ::Vector) where {ID,D}

  FEM_path(var::String) = joinpath(FEMInfo.Paths.FEM_structures_path, var)

  function FEM_vectors()
    Vecs = assemble_affine_FEM_vectors(FEMSpace, FEMInfo, μ, FEMInfo.t₀)
    FEM_paths = Broadcasting(FEM_path)(get_affine_vectors(FEMInfo))
    save_CSV(Vecs, FEM_paths)
  end

  function FEM_matrices()
    Mats = assemble_affine_FEM_matrices(FEMSpace, FEMInfo, μ, FEMInfo.t₀)
    FEM_paths = Broadcasting(FEM_path)(get_affine_matrices(FEMInfo))
    save_CSV(Mats, FEM_paths)
  end

  function FEM_norm_matrices()
    norm_operators = "X".*FEMInfo.unknowns
    norm_Mats = assemble_FEM_matrix(FEMSpace, FEMInfo, μ, norm_operators, FEMInfo.t₀)
    FEM_paths = Broadcasting(FEM_path)("X".*FEMInfo.unknowns)
    save_CSV(norm_Mats, FEM_paths)
  end

  FEM_vectors()
  FEM_matrices()
  FEM_norm_matrices()

  return

end

function run_param_loop(
  FEMInfo::FOMInfo{ID},
  μ::Vector{Vector{Float}}) where ID

  FEMSpace, FEMμ = get_FEM_solution(FEMInfo)

  get_FEM_structures(FEMSpace, FEMInfo, μ[1])
  function FEMμₖ(k::Int)
    println("Collecting solution number $k")
    FEMμ(μ[k])
  end

  xₕ_block = Broadcasting(FEMμₖ)(1:FEMInfo.nₛ)
  uₕ_block, pₕ_block = first.(xₕ_block), last.(xₕ_block)
  blocks_to_matrix(uₕ_block), blocks_to_matrix(pₕ_block)

end

function get_FEM_results(FEMInfo::FOMInfo{ID}) where ID

  _, μfun = config_FEM()
  μ = μfun(FEMInfo.nₛ)::Vector{Vector{Float}}

  FEM_time = @elapsed begin
    uₕ, pₕ = run_param_loop(FEMInfo, μ)
  end

  save_structures_in_list(
    (uₕ, pₕ, μ, [FEM_time/FEMInfo.nₛ]),
    ("uₕ", "pₕ", "μ", "FEM_time"),
    FEMInfo.Paths.FEM_snap_path)

end

get_FEM_results(FEMInfo)
