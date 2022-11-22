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
    matvec = ["A", "F", "H", "LA"]
  elseif name == "stokes"
    matvec = ["A", "B", "F", "H", "LA", "LB"]
  elseif name == "navier-stokes"
    matvec = ["A", "B", "C", "D", "F", "H", "LA", "LB", "LC"]
  else
    error("Not implemented")
  end
  issteady ? matvec : push!(matvec, "M")
end

function get_FEM_structures(FEMInfo::FOMInfo{ID}) where ID
  if ID == 1
    matvec = ["A", "F", "H", "LA"]
  elseif ID == 2
    matvec = ["A", "B", "F", "H", "LA", "LB"]
  else ID == 3
    matvec = ["A", "B", "C", "D", "F", "H", "LA", "LB", "LC"]
  end
  (typeof(FEMInfo) <: FOMInfoST{ID}) ? matvec : vcat(matvec, ["M", "LM"])
end

function get_FEM_vectors(FEMInfo::FOMInfo{ID}) where ID
  vecs = ["F", "H", "LA", "LB", "LC"]
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

function add_affine_lifts(affine_structures::Vector{String})
  if "L" ∈ affine_structures
    aff_structs = setdiff(affine_structures, "L")
    aff_mats = intersect(aff_structs, ["A", "B"])
    vcat(aff_structs, "L" .* aff_mats)
  else
    affine_structures
  end
end

function get_μ(FEMInfo::FOMInfo{ID}) where ID
  load_CSV(Vector{Float}[],
    joinpath(FEMInfo.Paths.snap_path, "μ.csv"))::Vector{Vector{Float}}
end

function get_FEMμ_info(FEMInfo::FOMInfoS{ID}, ::Val{D}) where {ID,D}
  μ = get_μ(FEMInfo)::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{D,D}
  FEMSpace₀ = get_FEMSpace(FEMInfo, model)::FOMS{ID,D}

  FEMSpace₀, μ

end

function get_FEMμ_info(FEMInfo::FOMInfoST{ID}, ::Val{D}) where {ID,D}
  μ = get_μ(FEMInfo)::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{D,D}
  FEMSpace₀ = get_FEMSpace(FEMInfo, model)::FOMST{ID,FEMInfo.D}

  FEMSpace₀, μ

end

function get_FEMμ_info(FEMInfo::FOMInfoS{ID}, μ::Vector{T}, ::Val{D}) where {ID,D,T}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{D,D}
  get_FEMSpace(FEMInfo, model, μ)::FOMS{ID,D}
end

function get_FEMμ_info(FEMInfo::FOMInfoST{ID}, μ::Vector{T}, ::Val{D}) where {ID,D,T}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)::DiscreteModel{D,D}
  get_FEMSpace(FEMInfo, model, μ)::FOMST{ID,D}
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

function get_g₁(::FOMInfoS{1})
  x -> 1.
end

function get_g₁(::FOMInfoST{1})
  g₁(x, t::Real) = 1.
  g₁(t::Real) = x -> g₁(x, t)
  g₁
end

function get_g₁(FEMInfo::FOMInfoS{2})
  x -> one(VectorValue(FEMInfo.D, Float))
end

function get_g₁(FEMInfo::FOMInfoST{2})
  g₁(x, t::Real) = one(VectorValue(FEMInfo.D, Float))
  g₁(t::Real) = x -> g₁(x, t)
  g₁
end

function get_g₁(FEMInfo::FOMInfoS{3})
  x -> one(VectorValue(FEMInfo.D, Float))
end

function get_g₁(FEMInfo::FOMInfoST{3})
  g₁(x, t::Real) = one(VectorValue(FEMInfo.D, Float))
  g₁(t::Real) = x -> g₁(x, t)
  g₁
end

function get_∂g(FEMSpace::FOM{ID,D}, g::Function) where {ID,D}
  ∂g_all(x,t::Real) = ∂t(g)(x,t)
  ∂g_all(t::Real) = x -> ∂g_all(x,t)
  ∂g(t) = interpolate_dirichlet(∂g_all(t), FEMSpace.V[1](t))
  ∂g
end

function get_h(FEMSpace::FOM{ID,D}) where {ID,D}
  Λ = SkeletonTriangulation(FEMSpace.Ω)
  dΛ = Measure(Λ, 2)
  h = get_array(∫(1)dΛ)[1]
  h
end

function dirichlet_dofs_on_full_trian(V, V_no_bc)

  cell_dof_ids_V = get_cell_dof_ids(V)
  cell_dof_ids_V_no_bc = get_cell_dof_ids(V_no_bc)

  dirichlet_dofs = zeros(Int, V.ndirichlet)
  for cell=eachindex(cell_dof_ids_V)
    for (idsV,idsV_no_bc) in zip(cell_dof_ids_V[cell],cell_dof_ids_V_no_bc[cell])
      if idsV<0
        dirichlet_dofs[abs(idsV)]=idsV_no_bc
      end
    end
  end

  dirichlet_dofs

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
