include("FunctionDefinitions.jl")

function fem_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function rom_path(tpath::String,ϵ::Float)
  rbpath = joinpath(tpath,"$ϵ")
  create_dir!(rbpath)
  rbpath
end

function rom_offline_path(tpath::String,ϵ::Float)
  rb_off_path = joinpath(rom_path(tpath,ϵ),"offline")
  create_dir!(rb_off_path)
  rb_off_path
end

function rom_online_path(tpath::String,ϵ::Float)
  rb_on_path = joinpath(rom_path(tpath,ϵ),"online")
  create_dir!(rb_on_path)
  rb_on_path
end

function rom_off_on_paths(
  tpath::String,ϵ::Float;
  st_mdeim=false,fun_mdeim=false)

  @assert isdir(tpath) "Provide valid path for the current test"
  function keyword()
    if !st_mdeim && !fun_mdeim
      return "standard"
    else
      st = st_mdeim ? "st" : ""
      fun = fun_mdeim ? "fun" : ""
      return st*fun
    end
  end

  rompath = joinpath(tpath,"rom")
  keytpath = joinpath(rompath,keyword())

  offpath = rom_offline_path(keytpath,ϵ)
  onpath = rom_online_path(keytpath,ϵ)
  offpath,onpath
end

function mesh_path(tpath::String,mesh::String)
  joinpath(get_parent_dir(tpath;nparent=3),"meshes/$mesh")
end

function set_labels!(model,bnd_info)
  tags = collect(keys(bnd_info))
  bnds = collect(values(bnd_info))
  @assert length(tags) == length(bnds)
  labels = get_face_labeling(model)
  for i = eachindex(tags)
    if tags[i] ∉ labels.tag_to_name
      add_tag_from_tags!(labels,tags[i],bnds[i])
    end
  end
end

function model_info(
  mshpath::String,
  bnd_info::Dict)

  if !ispath(mshpath)
    mshpath_msh_format = mshpath[1:findall(x->x=='.',mshpath)[end]-1]*".msh"
    model_msh_format = GmshDiscreteModel(mshpath_msh_format)
    to_json_file(model_msh_format,mshpath)
  end
  model = DiscreteModelFromFile(mshpath)
  set_labels!(model,bnd_info)
  model
end

function generate_fe_snapshots(
  isindef::Val,
  run_fem::Bool,
  fepath::String,
  nsnap::Int,
  solver,
  op,
  args...;
  save_snaps=true)

  if run_fem
    printstyled("Generating $nsnap full order snapshots on each available worker\n";color=:blue)
    fe_time = @elapsed begin
      sols_params = generate_fe_snapshots(isindef,solver,op,nsnap,args...)
    end
    printstyled("fe_time = $fe_time\n";color=:blue)
    if save_snaps
      save(fepath,sols_params)
      save(joinpath(fepath,"fe_time"),fe_time/ceil(nsnap/nworkers()))
    end
  else
    sols_params = load(isindef,fepath,nsnap)
  end

  sols_params
end

function load(
  ::Val{false},
  fepath::String,
  nsnap::Int)

  load(fepath,:u,nsnap),load(Vector{Param},fepath)
end

function load(
  ::Val{true},
  fepath::String,
  nsnap::Int)

  load(fepath,:u,nsnap),load(fepath,:p,nsnap),load(Vector{Param},fepath)
end

function generate_fe_snapshots(
  isindef::Val,
  solver::FESolver,
  op::ParamFEOperator,
  nsnap::Int)

  sol = solve(solver,op,nsnap)
  generate_fe_snapshots(isindef,sol)
end

function generate_fe_snapshots(
  isindef::Val,
  solver::ThetaMethod,
  op::ParamTransientFEOperator,
  nsnap::Int,
  t0::Real,
  tF::Real)

  sol = solve(solver,op,t0,tF,nsnap)
  generate_fe_snapshots(isindef,sol)
end

function generate_fe_snapshots(
  ::Val{false},
  sol)

  ns = length(sol)
  xh,μ = get_solutions(sol)
  usnap = Snapshots(:u,xh,ns)
  usnap,μ
end

function generate_fe_snapshots(
  ::Val{true},
  sol)

  Ns,ns = get_Ns(sol),length(sol)
  xh,μ = get_solutions(sol)
  u,p = xh[1:Ns[1],:],xh[Ns[1]+1:Ns[1]+Ns[2],:]
  usnap,psnap = Snapshots(:u,u,ns),Snapshots(:p,p,ns)
  usnap,psnap,μ
end

function get_solutions(sol)
  Ns = sum(get_Ns(first(sol)))
  Nt = get_Nt(first(sol))
  np = get_np(first(sol))
  x = allocate_matrix(Matrix{Float},Ns,Nt)
  μ = zeros(np)

  idx = eachindex(sol)
  sols_and_params = pmap(k->get_solution!(x,μ,sol,k),idx)
  sols,params = first.(sols_and_params),last.(sols_and_params)
  EMatrix(sols),vector_of_params(params)
end

function get_solution!(
  x::Matrix{Float},
  μ::Vector{Float},
  sol::Vector{ParamFESolution},
  k::Int)

  printstyled("Computing snapshot $k\n";color=:blue)

  solk = sol[k]
  copyto!(view(x,:,1),get_free_dof_values(solk.psol.uh))
  copyto!(μ,get_μ(solk.psol.μ))

  printstyled("Successfully computed snapshot $k\n";color=:blue)

  x,μ
end

function get_solution!(
  x::Matrix{Float},
  μ::Vector{Float},
  sol::Vector{ParamTransientFESolution},
  k::Int)

  printstyled("Computing snapshot $k\n";color=:blue)

  solk = sol[k]
  n = 1
  for (xn,_) in solk
    copyto!(view(x,:,n),xn)
    n += 1
  end
  copyto!(μ,get_μ(solk.psol.μ))

  printstyled("Successfully computed snapshot $k\n";color=:blue)

  x,μ
end

function get_dirichlet_values(
  U::ParamTrialFESpace,
  μ::Vector{Param})

  nsnap = length(μ)
  dir(μ) = U(μ).dirichlet_values
  Snapshots(:g,dir.(μ),nsnap,EMatrix{Float})
end

function get_dirichlet_values(
  U::ParamTransientTrialFESpace,
  μ::Vector{Param},
  tinfo::TimeInfo)

  nsnap = length(μ)
  times = get_times(tinfo)
  dir(μ) = Matrix([U(μ,t).dirichlet_values for t=times])
  Snapshots(:g,dir.(μ),nsnap,EMatrix{Float})
end

function online_loop(fe_sol,rb_space,rb_system,k::Int)
  online_time = @elapsed begin
    lhs,rhs = rb_system(k)
    rb_sol = solve_rb_system(lhs,rhs)
  end
  fe_sol_approx = reconstruct_fe_sol(rb_space,rb_sol)

  RBResults(fe_sol,fe_sol_approx,online_time)
end

function online_loop(loop,k::UnitRange{Int})
  RBResults(pmap(loop,k))
end
