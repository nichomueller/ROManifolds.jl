function setup_FEM(name::String, issteady::Bool)
  (get_id(name, issteady), get_unknowns(name),
    get_FEM_structures(name, issteady))
end

function get_id(name::String, issteady::Bool)
  if name == "poisson"
    issteady ? (0,) : (0,0)
  elseif name == "stokes"
    issteady ? (0,0,0) : (0,0,0,0)
  elseif name == "navier_stokes"
    issteady ? (0,0,0,0,0) : (0,0,0,0,0,0)
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
  if !issteady push!(matvec, ["M"]) end
end

function get_FEM_structures(NT::NTuple)
  if typeof(NT) ∈ (NTuple{1, Int}, NTuple{2, Int})
    ["A", "F", "H", "L"]
  elseif typeof(NT) ∈ (NTuple{3, Int}, NTuple{4, Int})
    ["A", "B", "F", "H", "L", "Lc"]
  else typeof(NT) ∈ (NTuple{5, Int}, NTuple{6, Int})
    ["A", "B", "C", "D", "F", "H", "L", "Lc"]
  end
end

function get_FEM_structures(FEMInfo::FOMInfoS)
  get_FEM_structures(FEMInfo.id)
end

function get_FEM_structures(FEMInfo::FOMInfoST)
  append!(["M"], get_FEM_structures(FEMInfo.id))
end

function get_FEM_vectors(FEMInfo::FOMInfo)
  vecs = ["F", "H", "L", "Lc"]
  intersect(FEMInfo.structures, vecs)::Vector{String}
end

function get_FEM_matrices(FEMInfo::FOMInfo)
  setdiff(FEMInfo.structures, get_FEM_vectors(FEMInfo))::Vector{String}
end

function get_FEMμ_info(FEMInfo::FOMInfo)
  μ = load_CSV(Vector{Float}[],
    joinpath(FEMInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{Float}}
  model = DiscreteModelFromFile(FEMInfo.Paths.mesh_path)
  FEMSpace = FEMSpace₀(FEMInfo.id, FEMInfo, model)::FOM

  FEMSpace, μ

end

function get_h(FEMSpace::Problem)
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

function set_labels(
  model::DiscreteModel,
  bnd_info::Dict)

  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)

  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels, tags[i], bnds[i])
    end
  end

end
