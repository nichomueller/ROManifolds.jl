struct OfflineTime
  basis_time::Float
  assembly_time::Float
end

function save(info::RBInfo,ot::OfflineTime)
  ot_dict = Dict("basis_time"=>ot.basis_time,"assembly_time"=>ot.assembly_time)
  save(joinpath(info.offline_path,"offline_time"),ot_dict)
end

struct RBResults
  sol::Snapshots
  sol_approx::Snapshots
  relative_err::Float
  online_time::Float
end

function RBResults(
  sol::NTuple{N,Snapshots},
  sol_approx::NTuple{N,Snapshots},
  args...;
  kwargs...)::NTuple{N,RBResults} where N

  RBResults.(sol,sol_approx,args...;kwargs...)
end

function RBResults(
  sol::Snapshots,
  sol_approx::Snapshots,
  online_time::Float;
  X=nothing)

  sol_mat = get_snap(sol)
  sol_approx_mat = get_snap(sol_approx)
  Y = isnothing(X) ? I(size(sol_mat,1)) : X
  relative_err = compute_rel_error(sol_mat,sol_approx_mat,Y)
  RBResults(sol,sol_approx,relative_err,online_time)
end

function RBResults(res::Vector{RBResults})
  nruns = length(res)

  sols = first(res).sol
  sols_approx = first(res).sol_approx
  relative_errs = Broadcasting(r->getproperty(r,:relative_err))(res)
  online_times = Broadcasting(r->getproperty(r,:online_time))(res)

  relative_err = sum(relative_errs)/nruns
  online_time = sum(online_times)/nruns

  RBResults(sols,sols_approx,relative_err,online_time)
end

function RBResults(res::Vector{NTuple{2,RBResults}})
  RBResults(first.(res)),RBResults(last.(res))
end

function save(info::RBInfo,res::NTuple{1,RBResults})
  res_u, = res
  res_dict = Dict("relative_err_$(res_u.sol.id)" => res_u.relative_err,
                  "online_time"=>res_u.online_time)
  save(joinpath(info.online_path,"results"),res_dict)
end

function save(info::RBInfo,res::NTuple{2,RBResults})
  res_u,res_p = res
  res_dict = Dict("relative_err_$(res_u.sol.id)" => res_u.relative_err,
                  "relative_err_$(res_p.sol.id)" => res_p.relative_err,
                  "online_time"=>res_u.online_time)
  save(joinpath(info.online_path,"results"),res_dict)
end

function compute_rel_error(
  u::AbstractMatrix,
  uh_rb::AbstractMatrix,
  X::AbstractMatrix)

  Nt = size(u,2)
  absolute_err,uh_norm = zeros(Nt),zeros(Nt)
  for i = 1:Nt
    absolute_err[i] = sqrt((u[:,i]-uh_rb[:,i])'*X*(u[:,i]-uh_rb[:,i]))
    uh_norm[i] = sqrt(u[:,i]'*X*u[:,i])
  end

  norm(absolute_err)/norm(uh_norm)
end

function postprocess(
  info::RBInfo,
  res::NTuple{1,RBResults},
  fespaces::NTuple{1,Tuple},
  model::DiscreteModel,
  args...)

  res_u, = res
  fespaces_u, = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors err_u: $(res_u.relative_err)\n";
    color=:red)
  printstyled("Average online wall time: $(res_u.online_time) s\n";
    color=:red)
  printstyled("-------------------------------------------------------------\n")

  if info.save_online save(info,res) end

  if info.postprocess
    trian = get_triangulation(model)
    writevtk(info,res_u,fespaces_u,trian,args...)
  end

  return
end

function postprocess(
  info::RBInfo,
  res::NTuple{2,RBResults},
  fespaces::NTuple{2,Tuple},
  model::DiscreteModel,
  args...)

  res_u,res_p = res
  rel_err_u,rel_err_p = res_u.relative_err,res_p.relative_err
  fespaces_u,fespaces_p = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors (err_u,err_p): $((rel_err_u,rel_err_p))\n";
    color=:red)
  printstyled("Average online wall time: $(res_u.online_time) s\n";
    color=:red)
  printstyled("-------------------------------------------------------------\n")

  if info.save_online save(info,res) end

  if info.postprocess
    trian = get_triangulation(model)
    writevtk(info,res_u,fespaces_u,trian,args...)
    writevtk(info,res_p,fespaces_p,trian,args...)
  end

  return
end

function Gridap.writevtk(
  info::RBInfoSteady,
  s::Snapshots,
  fespace,
  trian::Triangulation)

  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  path = joinpath(plt_dir,"$(id)h")
  fefun = FEFunction(fespace,s.snap[:,1])
  writevtk(trian,path,cellfields=["$(id)"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoSteady,
  res::RBResults,
  fespaces::Tuple,
  trian::Triangulation)

  trial,test = fespaces
  sol,sol_approx = res.sol,res.sol_approx
  id = get_id(sol_approx)

  writevtk(info,sol,trial,trian)
  writevtk(info,sol_approx,trial,trian)

  plt_dir = joinpath(info.online_path,joinpath("plots","err_$(id)"))
  create_dir!(plt_dir)
  fefun = FEFunction(test,res.pointwise_err[:,1])
  writevtk(trian,plt_dir,cellfields=["err"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  s::Snapshots,
  fespace,
  trian::Triangulation,
  tinfo::TimeInfo)

  times = get_times(tinfo)
  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  for (it,t) in enumerate(times)
    fefun = FEFunction(fespace(t),Vector(s.snap[:,it]))
    writevtk(trian,joinpath(plt_dir,"$(id)_$(it).vtu"),cellfields=["$(id)"=>fefun])
  end
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  res::RBResults,
  fespaces::Tuple,
  trian::Triangulation,
  tinfo::TimeInfo)

  trial,test = fespaces
  times = get_times(tinfo)
  sol,sol_approx = res.sol,res.sol_approx
  writevtk(info,sol,trial,trian,tinfo)
  writevtk(info,sol_approx,trial,trian,tinfo)

  pointwise_err = abs.(get_snap(sol)-get_snap(sol_approx))
  id = get_id(sol_approx)

  plt_dir = joinpath(info.online_path,"plots")
  create_dir!(plt_dir)
  for it in eachindex(times)
    fefun = FEFunction(test,pointwise_err[:,it])
    writevtk(trian,joinpath(plt_dir,"err_$(id)_$(it).vtu"),cellfields=["err"=>fefun])
  end
end

function offline_results_dict(info::RBInfoUnsteady)
  dict = Dict("")
  tests_path = get_parent_dir(info.offline_path;nparent=3)
  summary_path = joinpath(tests_path,"results_summary")
  create_dir!(summary_path)

  for tpath in get_all_subdirectories(tests_path)
    if tpath != summary_path
      case = last(split(tpath,'/'))
      ns,nt,toff = (),(),""
      tolpath = first(get_all_subdirectories(tpath))
      offpath = joinpath(tolpath,"offline")
      for varpath in get_all_subdirectories(offpath)
        var = last(split(varpath,'/'))
        bspath = joinpath(varpath,"basis_space")
        btpath = joinpath(varpath,"basis_time")
        if myisfile(bspath) && myisfile(btpath)
          ns = (ns...,(var,size(load(bspath),2)))
          nt = (nt...,(var,size(load(btpath),2)))
        end
      end
      onpath = joinpath(tolpath,"online")
      if myisfile(joinpath(onpath,"times"))
        toff = prod(load(joinpath(onpath,"times"))[1,2:3])
      end
      dict = merge(dict,Dict("ns_$(case)"=>ns,"nt_$(case)"=>nt,
        "toff_$(case)"=>toff))
    end
  end

  save(joinpath(summary_path,"offline_results"),dict)
end

function online_results_dict(info::RBInfo)
  dict = Dict("")
  tests_path = get_parent_dir(info.offline_path;nparent=3)
  summary_path = joinpath(tests_path,"results_summary")
  create_dir!(summary_path)

  for tpath in get_all_subdirectories(tests_path)
    if tpath != summary_path
      case = last(split(tpath,'/'))
      err_case_u,err_case_p,ton_case = NaN,NaN,NaN
      for tolpath in get_all_subdirectories(tpath)
        tol = parse(Float,last(split(tolpath,'/')))
        onpath = joinpath(tolpath,"online")
        if myisfile(joinpath(onpath,"errors_u"))
          err_case_u = last(load(joinpath(onpath,"errors_u")))
        end
        if myisfile(joinpath(onpath,"errors_p"))
          err_case_p = last(load(joinpath(onpath,"errors_p")))
        end
        if myisfile(joinpath(onpath,"times"))
          ton = load(joinpath(onpath,"times"))[2,2]
          ton_case = ton
        end
        dict = merge(dict,Dict("erru_$(case)_$tol"=>err_case_u,
        "errp_$(case)_$tol"=>err_case_p,"ton_$(case)_$tol"=>ton_case))
      end
    end
  end

  save(joinpath(summary_path,"online_results"),dict)
end

function H1_norm_matrix(opA::ParamBilinOperator,opM::ParamBilinOperator)
  @assert get_id(opA) == :A && get_id(opM) == :M "wrong operators"
  assemble_affine_quantity(opA) + assemble_affine_quantity(opM)
end

function L2_norm_matrix(opM::ParamBilinOperator)
  @assert get_id(opM) == :M "wrong operator"
  assemble_affine_quantity(opM)
end

function gather_online_results(ptype,test_path)
  d = []
  for fun_mdeim=(true,), st_mdeim=(false,true), ϵ=(1e-2,1e-3,1e-4)
    info = RBInfoUnsteady(ptype,test_path;ϵ,nsnap=80,mdeim_snap=20,st_mdeim,fun_mdeim)
    tpath = info.online_path
    d = [d...,Dict("res_$(fun_mdeim)_$(st_mdeim)_$(ϵ)" => deserialize(joinpath(tpath,"results.txt")))]
  end
  d
end
