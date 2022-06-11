function FEM_paths(root, problem_type, problem_name, mesh_name, case; test_case="")

  @assert isdir(root) "$root is an invalid root directory"

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, problem_type)
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

function get_ParamInfo(::NTuple{1,Int},Info::SteadyInfo,μ::Vector)

  model = DiscreteModelFromFile(Info.paths.mesh_path)

  function prepare_α(x, μ, probl_nl)
    if !probl_nl["A"]
      return sum(μ)
    else
      return 1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    end
  end
  α(x) = prepare_α(x, μ, Info.probl_nl)

  function prepare_f(x, μ, probl_nl)
    if probl_nl["f"]
      return sin(μ[4] * x[1]) + sin(μ[4] * x[2])
    else
      return 1
    end
  end
  f(x) = prepare_f(x, μ, Info.probl_nl)
  g(x) = 0
  h(x) = 1

  ParametricInfo(μ, model, α, f, g, h)

end

function get_ParamInfo(::NTuple{1,Int},Info::UnsteadyInfo,μ::Vector)

  model = DiscreteModelFromFile(Info.paths.mesh_path)
  αₛ(x) = 1
  αₜ(t, μ) = sum(μ) * (2 + sin(2π * t))
  mₛ(x) = 1
  mₜ(t::Real) = 1
  m(x, t::Real) = mₛ(x)*mₜ(t)
  m(t::Real) = x -> m(x, t)
  fₛ(x) = 1
  fₜ(t::Real) = sin(π * t)
  gₛ(x) = 0
  gₜ(t::Real) = 0
  g(x, t::Real) = gₛ(x)*gₜ(t)
  g(t::Real) = x -> g(x, t)
  hₛ(x) = 0
  hₜ(t::Real) = 0
  h(x, t::Real) = hₛ(x)*hₜ(t)
  h(t::Real) = x -> h(x, t)
  u₀(x) = 0

  function prepare_α(x, t, μ, probl_nl)
    if !probl_nl["A"]
      return αₛ(x)*αₜ(t, μ)
    else
      return (10 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    end
  end
  α(x, t::Real) = prepare_α(x, t, μ, Info.probl_nl)
  α(t::Real) = x -> α(x, t)

  function prepare_f(x, t, μ, probl_nl)
    if !probl_nl["f"]
      return fₛ(x)*fₜ(t)
    else
      return 10+5*sin(π*t*sum(x)*(μ[4]+μ[5]))
    end
  end
  f(x, t::Real) = prepare_f(x, t, μ, Info.probl_nl)
  f(t::Real) = x -> f(x, t)

  ParametricInfoUnsteady(
    μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, g, hₛ, hₜ, h, u₀)

end

function get_ParamInfo(::NTuple{2,Int},Info::UnsteadyInfo,μ::Vector)

  model = DiscreteModelFromFile(Info.paths.mesh_path)
  αₛ(x) = 1
  αₜ(t::Real, μ) = sum(μ) * (2 + sin(2π * t))
  mₛ(x) = 1
  mₜ(t::Real) = 1
  m(x, t::Real) = mₛ(x)*mₜ(t)
  m(t::Real) = x -> m(x, t)
  fₛ(x) = VectorValue(0.,0.,0.)
  fₜ(t::Real) = 0
  f(x, t::Real) = fₛ(x)*fₜ(t)
  f(t::Real) = x -> f(x, t)
  gʷ(x, t::Real) = VectorValue(0.,0.,0.)
  gʷ(t::Real) = x -> gʷ(x, t)
  x0 = Point(0.,0.,0.)
  R = 1
  gₛ(x) = 2 * (1 .- VectorValue((x[1]-x0[1])^2,(x[2]-x0[2])^2,(x[3]-x0[3])^2) / R^2) / (pi*R^2)
  gₜ(t::Real, μ) = 1-cos(2*pi*t/T)+μ[2]*sin(2*pi*μ[1]*t/T)
  gⁱⁿ(x, t::Real) = gₛ(x)*gₜ(t, μ)
  gⁱⁿ(t::Real) = x -> gⁱⁿ(x, t)
  hₛ(x) = VectorValue(0.,0.,0.)
  hₜ(t::Real) = 0
  h(x, t::Real) = hₛ(x)*hₜ(t)
  h(t::Real) = x -> h(x, t)
  u₀(x) = VectorValue(0.,0.,0.)
  p₀(x) = 0.
  function x₀(x)
    return [u₀(x), p₀(x)]
  end

  function prepare_α(x, t, μ, probl_nl)
    if !probl_nl["A"]
      return αₛ(x)*αₜ(t, μ)
    else
      return (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    end
  end
  α(x, t::Real) = prepare_α(x, t, μ, Info.probl_nl)
  α(t::Real) = x -> α(x, t)

  function prepare_g(x, t, μ, case)
    if case <= 1
      gⁱⁿ(x, t::Real) = gₛ(x)*gₜ(t, μ)
      gⁱⁿ(t::Real) = x -> gⁱⁿ(x, t)
      return gⁱⁿ#[gʷ,gⁱⁿ]
    else
      gⁱⁿ₁(x, t::Real) = gₛ(x)*gₜ(t, μ[end-2:end-1])*μ[end]
      gⁱⁿ₁(t::Real) = x -> g(x, t)
      gⁱⁿ₂(x, t::Real) = gₛ(x)*(1-gₜ(t, μ[end-2:end-1])*μ[end])
      gⁱⁿ₂(t::Real) = x -> g(x, t)
      return gⁱⁿ₁#[gʷ,gⁱⁿ₁,gⁱⁿ₂]
    end
  end
  #g(x, t::Real) = prepare_g(x, t, μ, Info.case)
  g(x, t::Real) = gₛ(x)*gₜ(t, μ)
  g(t::Real) = x -> g(x, t)

  ParametricInfoUnsteady(
    μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, g, hₛ, hₜ, h, x₀)

end

function generate_vtk_file(FEMSpace::FEMProblem, path::String, var_name::String, var::Array)

  FE_var = FEFunction(FEMSpace.V, var)
  writevtk(FEMSpace.Ω, path, cellfields = [var_name => FE_var])

end

function find_FE_elements(V₀::UnconstrainedFESpace, trian::Triangulation, idx::Vector)

  connectivity = get_cell_dof_ids(V₀, trian)

  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(connectivity)[1]
      if idx[i] in abs.(connectivity[j])
        append!(el, j)
      end
    end
  end

  unique(el)

end

function generate_dcube_discrete_model(I::Info,d::Int64,npart::Int,mesh_name::String)

  mesh_dir = I.paths.mesh_path[1:findall(x->x=='/',I.paths.mesh_path)[end]]
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
