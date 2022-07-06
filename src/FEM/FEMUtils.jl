function FEM_paths(root, problem_steadiness, problem_name, mesh_name, case; test_case="")

  @assert isdir(root) "$root is an invalid root directory"

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, problem_steadiness)
  create_dir(type_path)
  problem_path = joinpath(type_path, problem_name)
  create_dir(problem_path)
  problem_and_info_path = joinpath(problem_path, "case" * string(case))
  create_dir(problem_and_info_path)
  current_test = joinpath(problem_and_info_path, mesh_name)
  create_dir(current_test)
  FEM_path = joinpath(current_test, "FEM_data")
  create_dir(FEM_path)
  FEM_snap_path = joinpath(FEM_path, "snapshots"*test_case)
  create_dir(FEM_snap_path)
  FEM_structures_path = joinpath(FEM_path, "FEM_structures")
  create_dir(FEM_structures_path)

  _ -> (mesh_path, current_test, FEM_snap_path, FEM_structures_path)

end

function get_problem_id(problem_name::String)
  if problem_name == "poisson"
    return (0,)
  elseif problem_name == "stokes"
    return (0,0)
  elseif problem_name == "navier-stokes"
    return (0,0,0)
  else
    error("unimplemented")
  end
end

function init_FEM_variables(
  ::Info{T}) where T

  M = sparse([], [], T[])
  A = sparse([], [], T[])
  B = sparse([], [], T[])
  Xᵘ₀ = sparse([], [], T[])
  Xᵘ = sparse([], [], T[])
  Xᵖ₀ = sparse([], [], T[])
  F = Vector{T}(undef,0)
  H = Vector{T}(undef,0)

  M, A, B, Xᵘ₀, Xᵘ, Xᵖ₀, F, H

end

function get_ParamInfo(
  FEMInfo::SteadyInfo{T},
  problem_id::NTuple{1,Int},
  μ::Vector) where T

  α(x) = get_α(FEMInfo, problem_id, μ).α(x)
  f(x) = get_f(FEMInfo, problem_id, μ).f(x)
  g(x) = get_g(FEMInfo, problem_id, μ).g(x)
  h(x) = get_h(FEMInfo, problem_id, μ).h(x)

  ParametricInfoSteady(μ, α, f, g, h)

end

function get_α(FEMInfo::SteadyInfo{T}, ::NTuple{1,Int64}, μ) where T
  if !FEMInfo.probl_nl["A"]
    return T(sum(μ))
  else
    return T(1. + μ[3] + 1. / μ[3] * exp(-norm(x-Point(μ[1:FEMInfo.D]))^2 / μ[3]))
  end
end

function get_f(FEMInfo::SteadyInfo{T}, ::NTuple{1,Int64}, μ) where T
  if !FEMInfo.probl_nl["f"]
    return one(T)
  else
    return T(1 + sin(norm(Point(μ[4:3+FEMInfo.D]) .* x)*t))
  end
end

function get_g(FEMInfo::SteadyInfo{T}, ::NTuple{1,Int64}, μ) where T
  if !FEMInfo.probl_nl["g"]
    return zero(T)
  else
    return zero(T)
  end
end

function get_h(FEMInfo::SteadyInfo{T}, ::NTuple{1,Int64}, μ) where T
  if FEMInfo.probl_nl["h"]
    return 1 + sin(Point(μ[end-FEMInfo.D+1:end]) .* x)
  else
    return one(T)
  end
end

function get_ParamInfo(
  FEMInfo::UnsteadyInfo{T},
  ::NTuple{1,Int},
  μ::Vector) where T

  αₛ(x) = one(T)
  αₜ(t, μ) = T(5. * sum(μ) * (2 + sin(t)))
  function α(x, t::Real)
    if !FEMInfo.probl_nl["A"]
      return T(αₛ(x)*αₜ(t, μ))
    else
      return T(10. * (1. + 1. / μ[3] * exp(-norm(x-Point(μ[1:FEMInfo.D]))^2 * sin(t) / μ[3])))
    end
  end
  α(t::Real) = x -> α(x, t)

  fₛ(x) = one(T)
  fₜ(t::Real) = T(sin(t))
  function f(x, t::Real)
    if !FEMInfo.probl_nl["f"]
      return T(fₛ(x)*fₜ(t))
    else
      return T(1 + sin(norm(Point(μ[4:3+FEMInfo.D]) .* x)*t))
    end
  end
  f(t::Real) = x -> f(x, t)

  gₛ(x) = zero(T)
  gₜ(t::Real) = zero(T)
  function g(x, t::Real)
    if !FEMInfo.probl_nl["g"]
      return T(gₛ(x)*gₜ(t))
    else
      return zero(T)
    end
  end
  g(t::Real) = x -> g(x, t)

  hₛ(x) = one(T)
  hₜ(t::Real) = T(sin(t))
  function h(x, t::Real)
    if !FEMInfo.probl_nl["h"]
      return T(hₛ(x)*hₜ(t))
    else
      return T(1 + sin(norm(Point(μ[end-FEMInfo.D+1:end]) .* x)*t))
    end
  end
  h(t::Real) = x -> h(x, t)

  mₛ(x) = one(T)
  mₜ(t::Real) = one(T)
  function m(x, t)
    if !FEMInfo.probl_nl["M"]
      return T(mₛ(x)*mₜ(t))
    else
      return one(T)
    end
  end
  m(t::Real) = x -> m(x, t)

  u₀(x) = zero(T)

  ParametricInfoUnsteady{T}(
    μ, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)

end

function get_ParamInfo(
  FEMInfo::UnsteadyInfo{T},
  ::NTuple{2,Int},
  μ::Vector) where T

  αₛ(x) = one(T)
  αₜ(t, μ) = T(5. * sum(μ) * (2 + sin(t)))
  function α(x, t::Real)
    if !FEMInfo.probl_nl["A"]
      return T(αₛ(x)*αₜ(t, μ))
    else
      return T(10. * (1. + 1. / μ[3] * exp(-norm(x-Point(μ[1:FEMInfo.D]))^2 * sin(t) / μ[3])))
    end
  end
  α(t::Real) = x -> α(x, t)

  fₛ(x) = one(VectorValue(FEMInfo.D, T))
  fₜ(t::Real) = one(T)
  function f(x, t)
    if !FEMInfo.probl_nl["f"]
      return T(fₛ(x)*fₜ(t))
    else
      return T(1 + Point(μ[4:3+FEMInfo.D]) .* x*t)
    end
  end

  gₛ(x) = zero(VectorValue(FEMInfo.D, T))
  gₜ(t::Real) = zero(T)
  function g(x, t::Real)
    if !FEMInfo.probl_nl["g"]
      return T(gₛ(x)*gₜ(t))
    else
      return zero(VectorValue(FEMInfo.D, T))
    end
  end
  g(t::Real) = x -> g(x, t)

  hₛ(x) = one(VectorValue(FEMInfo.D, T))
  hₜ(t::Real) = T(sin(t))
  function h(x, t::Real)
    if !FEMInfo.probl_nl["h"]
      return T(hₛ(x)*hₜ(t))
    else
      return T(1 + Point(μ[end-FEMInfo.D+1:end]) .* x*t)
    end
  end
  h(t::Real) = x -> h(x, t)

  mₛ(x) = one(T)
  mₜ(t::Real) = one(T)
  function m(x, t)
    if !FEMInfo.probl_nl["M"]
      return T(mₛ(x)*mₜ(t))
    else
      return one(T)
    end
  end
  m(t::Real) = x -> m(x, t)

  u₀(x) = [zero(VectorValue(FEMInfo.D, T)), zero(T)]

  ParametricInfoUnsteady{T}(
    μ, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)

end

function nonlinearity_lifting_op(FEMInfo::Info)
  if !FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    return 0
  elseif FEMInfo.probl_nl["A"] && !FEMInfo.probl_nl["g"]
    return 1
  elseif !FEMInfo.probl_nl["A"] && FEMInfo.probl_nl["g"]
    return 2
  else
    return 3
  end
end

#= function get_f(FEMInfo::UnsteadyInfo{T}, ::NTuple{2,Int64}, μ) where T

  fₛ(x) = zero(VectorValue(N,T))
  fₜ(t::Real) = zero(T)
  function f(x, t)
    if !FEMInfo.probl_nl["f"]
      return T(fₛ(x)*fₜ(t))
    else
      return zero(VectorValue(FEMInfo.D,T))
    end
  end
  _-> (fₛ,fₜ,f)

end

function get_g(FEMInfo::UnsteadyInfo{T}, ::NTuple{2,Int64}, μ) where T
  x0 = zero(VectorValue(N,T))
  diff = x - x0
  gₛ(x) = T(1 .- diff .* diff)
  gₜ(t::Real, μ) = T(1-cos(t)+μ[end]*sin(μ[end-1]))
  function g(x, t)
    if FEMInfo.probl_nl["g"]
      return T(gₛ(x)*gₜ(t, μ))
    else
      return T(gₛ(x)*gₜ(t, μ))
    end
  end
  _-> (gₛ,gₜ,g)
end

function get_h(FEMInfo::UnsteadyInfo{T}, ::NTuple{2,Int64}, μ) where T
  hₛ(x) = zero(VectorValue(FEMInfo.D,T))
  hₜ(t::Real) = zero(T)
  h(x, t::Real) = zero(VectorValue(FEMInfo.D,T))
  _-> (hₛ,hₜ,h)
end

function get_IC(FEMInfo::UnsteadyInfo{T}, ::NTuple{2,Int64}) where T
  u₀(x) = [zero(T),zero(VectorValue(FEMInfo.D,T))]
end =#

function get_timesθ(FEMInfo::UnsteadyInfo{T}) where T
  T.(collect(FEMInfo.t₀:FEMInfo.δt:FEMInfo.tₗ-FEMInfo.δt).+FEMInfo.δt*FEMInfo.θ)
end

function generate_vtk_file(
  FEMSpace::FEMProblem,
  path::String,
  var_name::String,
  var::Array)

  FE_var = FEFunction(FEMSpace.V, var)
  writevtk(FEMSpace.Ω, path, cellfields = [var_name => FE_var])

end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BodyFittedTriangulation,
  idx::Vector{T}) where T

  connectivity = get_cell_dof_ids(V₀, trian)::Table{Int32, Vector{Int32}, Vector{Int32}}

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, convert(T,j))
      end
    end
  end

  unique(el)

end

function find_FE_elements(
  V₀::UnconstrainedFESpace,
  trian::BoundaryTriangulation,
  idx::Vector{T}) where T

  connectivity = collect(get_cell_dof_ids(V₀, trian))::Vector{Vector{Int32}}

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, convert(T,j))
      end
    end
  end

  unique(el)

end

function generate_dcube_discrete_model(FEMInfo::Info,d::Int,npart::Int,mesh_name::String)

  if !occursin(".json",mesh_name)
    mesh_name *= ".json"
  end
  mesh_dir = FEMInfo.paths.mesh_path[1:findall(x->x=='/',FEMInfo.paths.mesh_path)[end]]
  mesh_path = joinpath(mesh_dir,mesh_name)
  generate_dcube_discrete_model(d,npart,mesh_path)

end

function generate_dcube_discrete_model(d::Int,npart::Int,path::String)
  @assert d ≤ 3 "Select d-dimensional domain, where d ≤ 3"
  if d == 1
    domain = (0,1)
    partition = (npart)
  elseif d == 2
    domain = (0,1,0,1)
    partition = (npart,npart)
  else
    domain = (0,1,0,1,0,1)
    partition = (npart,npart,npart)
  end
  model = CartesianDiscreteModel(domain,partition)
  to_json_file(model,path)
end
