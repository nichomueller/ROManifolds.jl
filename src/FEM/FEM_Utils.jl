function setup_FEM(name::String, issteady::Bool)
  get_unknowns(name), get_FEM_structures(name, issteady)
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
  elseif name == "navier-stokes"
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

function get_FEM_vectors(FEMInfo::FOMInfo{ID}) where ID
  vecs = ["F", "H", "L", "Lc"]
  intersect(FEMInfo.structures, vecs)::Vector{String}
end

function isvector(FEMInfo::FOMInfo{ID}, var::String) where ID
  var ∈ get_FEM_vectors(FEMInfo)
end

function get_FEM_matrices(FEMInfo::FOMInfo{ID}) where ID
  setdiff(FEMInfo.structures, get_FEM_vectors(FEMInfo))::Vector{String}
end

function ismatrix(FEMInfo::FOMInfo{ID}, var::String) where ID
  var ∈ get_FEM_matrices(FEMInfo)
end

function get_affine_vectors(FEMInfo::FOMInfo{ID}) where ID
  intersect(get_FEM_vectors(FEMInfo), FEMInfo.affine_structures)
end

function get_affine_matrices(FEMInfo::FOMInfo{ID}) where ID
  intersect(get_FEM_matrices(FEMInfo), FEMInfo.affine_structures)
end

function get_nonaffine_vectors(FEMInfo::FOMInfo{ID}) where ID
  setdiff(get_FEM_vectors(FEMInfo), FEMInfo.affine_structures)
end

function get_nonaffine_matrices(FEMInfo::FOMInfo{ID}) where ID
  setdiff(get_FEM_matrices(FEMInfo), FEMInfo.affine_structures)
end

function isaffine(FEMInfo::FOMInfo{ID}, var::String) where ID
  var ∈ FEMInfo.affine_structures
end

function isaffine(FEMInfo::FOMInfo{ID}, vars::Vector{String}) where ID
  Broadcasting(var->isaffine(FEMInfo, var))(vars)
end

function get_FEMμ_info(FEMInfo::FOMInfoS{ID}, ::Val{D}) where {ID,D}
  μ = load_CSV(Vector{Float}[],
    joinpath(FEMInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{Float}}

  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{D,D}
  FEMSpace₀ = get_FEMSpace(FEMInfo, model)::FOMS{ID,D}

  FEMSpace₀, μ

end

function get_FEMμ_info(FEMInfo::FOMInfoST{ID}) where ID
  μ = load_CSV(Vector{Float}[],
    joinpath(FEMInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)
  FEMSpace₀ = get_FEMSpace(FEMInfo, model)::FOMST{ID,FEMInfo.D}

  FEMSpace₀, μ

end

function get_g₀(::FOMInfoS{1})
  x -> 0.
end

function get_g₀(::FOMInfoST{1})
  g₀(x, t::Real) = 0.
  g₀(t::Real) = x -> g₀(x, t)
  g₀
end

function get_g₀(FEMInfo::FOMInfoS{2})
  x -> zero(VectorValue(FEMInfo.D, Float))
end

function get_g₀(FEMInfo::FOMInfoST{2})
  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  g₀
end

function get_g₀(FEMInfo::FOMInfoS{3})
  x -> zero(VectorValue(FEMInfo.D, Float))
end

function get_g₀(FEMInfo::FOMInfoST{3})
  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  g₀
end

function get_h(FEMSpace::FOM{ID,D}) where {ID,D}
  Λ = SkeletonTriangulation(FEMSpace.Ω)
  dΛ = Measure(Λ, 2)
  h = get_array(∫(1)dΛ)[1]
  h
end

function get_timesθ(FEMInfo::FOMInfoST{ID}) where ID
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
  FEMSpace::FOM{ID,D},
  idx::Vector,
  var::String) where {ID,D}

  space = get_FEMSpace_vector(FEMSpace, var)
  triang = Gridap.FESpaces.get_triangulation(FEMSpace, var)

  find_FE_elements(space, triang, unique(idx))

end
