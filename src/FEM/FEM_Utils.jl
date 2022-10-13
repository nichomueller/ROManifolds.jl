function setup_FEM(name::String, issteady::Bool)
  get_unknowns(name), get_FEM_structures(name, issteady)
end

function get_id(name::String)
  if name == "poisson"
    1
  elseif name == "stokes"
    2
  elseif name == "navier_stokes"
    3
  else
    error("Not implemented")
  end
end

function get_unknowns(name::String)
  if name == "poisson"
    ["u"]
  elseif occursin("stokes", name)
    ["u", "p"]
  else
    error("Not implemented")
  end
end

function get_FEM_structures(name::String, issteady::Bool)
  if name == "poisson"
    matvec = ["A", "F", "H", "L"]
  elseif name == "stokes"
    matvec = ["A", "B", "F", "H", "L", "Lc"]
  elseif name == "navier_stokes"
    matvec = ["A", "B", "C", "D", "F", "H", "L", "Lc"]
  else
    error("Not implemented")
  end
  issteady ? matvec : push!(matvec, ["M"])
end

function get_FEM_structures(::FOMInfoS{ID}) where ID
  if ID == 1
    ["A", "F", "H", "L"]
  elseif ID == 2
    ["A", "B", "F", "H", "L", "Lc"]
  else ID == 3
    ["A", "B", "C", "D", "F", "H", "L", "Lc"]
  end
end

function get_FEM_structures(FEMInfo::FOMInfo{ID}) where ID
  append!(["M"], get_FEM_structures(FEMInfo))
end

function get_FEM_vectors(FEMInfo::FOMInfo)
  vecs = ["F", "H", "L", "Lc"]
  intersect(FEMInfo.structures, vecs)::Vector{String}
end

function isvector(FEMInfo::FOMInfo, var::String)
  var ∈ get_FEM_vectors(FEMInfo)
end

function get_FEM_matrices(FEMInfo::FOMInfo)
  setdiff(FEMInfo.structures, get_FEM_vectors(FEMInfo))::Vector{String}
end

function ismatrix(FEMInfo::FOMInfo, var::String)
  var ∈ get_FEM_matrices(FEMInfo)
end

function get_affine_vectors(FEMInfo)
  intersect(get_FEM_vectors(FEMInfo), FEMInfo.affine_structures)
end

function get_affine_matrices(FEMInfo)
  intersect(get_FEM_matrices(FEMInfo), FEMInfo.affine_structures)
end

function isaffine(FEMInfo::FOMInfo, var::String)
  var ∈ FEMInfo.affine_structures
end

function get_FEMμ_info(FEMInfo::FOMInfo)
  μ = load_CSV(Vector{Float}[],
    joinpath(FEMInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)
  FEMSpace₀ = get_FEMSpace(FEMInfo, model)::FOM

  FEMSpace₀, μ

end

function get_g₀(::FOMInfoS)
  x -> 0.
end

function get_g₀(::FOMInfoST)
  g₀(x, t::Real) = 0.
  g₀(t::Real) = x -> g₀(x, t)
  g₀
end

function get_h(FEMSpace::FOM)
  Λ = SkeletonTriangulation(FEMSpace.Ω)
  dΛ = Measure(Λ, 2)
  h = get_array(∫(1)dΛ)[1]
  h
end

function get_timesθ(FEMInfo::FOMInfoST)
  collect(FEMInfo.t₀:FEMInfo.δt:FEMInfo.tₗ-FEMInfo.δt).+FEMInfo.δt*FEMInfo.θ
end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BodyFittedTriangulation,
  idx::Vector)

  connectivity = get_cell_dof_ids(V₀, trian)::Table{Int32, Vector{Int32}, Vector{Int32}}

  el = Int[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, Int(j))
      end
    end
  end

  unique(el)

end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BoundaryTriangulation,
  idx::Vector)

  connectivity = collect(get_cell_dof_ids(V₀, trian))::Vector{Vector{Int32}}

  el = Int[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, Int(j))
      end
    end
  end

  unique(el)

end

function find_FE_elements(
  FEMSpace::FOM,
  idx::Vector,
  var::String)

  find_FE_elements(
    FEMSpace_vectors(FEMSpace, var), get_measure(FEMSpace, var), unique(idx))

end
