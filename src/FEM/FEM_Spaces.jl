function get_mod_meas_quad(FEMInfo::FOMInfo, model::DiscreteModel)

  function set_labels(bnd_info::Dict)

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

  set_labels(FEMInfo.bnd_info)

  degree = 2 * FEMInfo.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=["neumann"])
  dΓn = Measure(Γn, degree)::Measure
  Qₕ = CellQuadrature(Ω, degree)::CellQuadrature

  Ω, Γn, Qₕ, dΩ, dΓn

end

function get_lagrangianQuad_info(
  FEMInfo::FOMInfo,
  model::DiscreteModel{Dc,Dp},
  Ω::BodyFittedTriangulation,
  Qₕ::CellQuadrature) where {Dc,Dp}

  ξₖ = get_cell_map(Ω)
  Qₕ_cell_point = get_cell_points(Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = collect(lazy_map(Gridap.evaluate,ξₖ,qₖ))::Vector{Vector{VectorValue{Dp,Float}}}
  refFE_quad = Gridap.ReferenceFE(lagrangianQuad,Float,FEMInfo.order)
  V₀_quad = TestFESpace(model,refFE_quad,conformity=:L2)

  phys_quadp, V₀_quad

end

function get_FEMSpace(
  FEMInfo::FOMInfoS{ID},
  model::DiscreteModel{D,D},
  args...) where {ID,D}

  FOMS(FEMInfo, model, args...)

end

function get_FEMSpace(
  FEMInfo::FOMInfoST{ID},
  model::DiscreteModel{D,D},
  args...) where {ID,D}

  FOMST(FEMInfo, model, args...)

end

function get_FEMSpace_vector(
  FEMSpace::FOM{ID,D},
  var::String) where {ID,D}

  if var == "Lc"
    @assert ID != 1 "Something is wrong with problem variables"
    FEMSpace.V₀[2]
  else
    FEMSpace.V₀[1]
  end
end

function get_FEMSpace_matrix(
  FEMSpace::FOM{ID,D},
  var::String) where {ID,D}

  if var == "B"
    @assert ID != 1 "Something is wrong with problem variables"
    FEMSpace.V[1], FEMSpace.V₀[2]
  elseif var == "Xp"
    @assert ID != 1 "Something is wrong with problem variables"
    FEMSpace.V[2], FEMSpace.V₀[2]
  else
    FEMSpace.V[1], FEMSpace.V₀[1]
  end
end

function get_measure(FEMSpace::FOM, var::String)
  if var == "H"
    FEMSpace.dΓn
  else
    FEMSpace.dΩ
  end
end

function Gridap.FESpaces.get_triangulation(FEMSpace::FOM, var::String)
  if var == "H"
    FEMSpace.Γn
  else
    FEMSpace.Ω
  end
end
