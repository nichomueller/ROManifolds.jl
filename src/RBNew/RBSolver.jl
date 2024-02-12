function get_test_dir(path::String,ϵ;st_mdeim=false)
  keyword = st_mdeim ? "st" : "standard"
  outer_path = joinpath(path,keyword)
  dir = joinpath(outer_path,"$ϵ")
  FEM.create_dir(dir)
  dir
end

struct SpaceOnlyMDEIM end
struct SpaceTimeMDEIM end

struct RBInfo{M}
  ϵ::Float64
  mdeim_style::M
  norm_style::Symbol
  dir::String
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  save_structures::Bool
end

function RBInfo(
  test_path::String;
  ϵ=1e-4,
  st_mdeim=false,
  norm_style=:l2,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  save_structures=true)

  mdeim_style = st_mdeim == true ? SpaceTimeMDEIM() : SpaceOnlyMDEIM()
  dir = get_test_dir(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,mdeim_style,norm_style,dir,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,save_structures)
end

num_offline_params(info::RBInfo) = info.nsnaps_state
offline_params(info::RBInfo) = 1:num_offline_params(info)
num_online_params(info::RBInfo) = info.nsnaps_test
online_params(info::RBInfo) = 1+num_offline_params(info):num_online_params(info)+num_offline_params(info)
FEM.num_params(info::RBInfo) = num_offline_params(info) + num_online_params(info)
num_mdeim_params(info::RBInfo) = info.nsnaps_mdeim
mdeim_params(info::RBInfo) = 1:num_offline_params(info)
get_tol(info::RBInfo) = info.ϵ

function get_norm_matrix(info::RBInfo,feop::TransientParamFEOperator)
  norm_style = info.norm_style
  try
    T = get_vector_type(feop.test)
    load(info,SparseMatrixCSC{eltype(T),Int};norm_style)
  catch
    if norm_style == :l2
      nothing
    elseif norm_style == :L2
      get_L2_norm_matrix(feop)
    elseif norm_style == :H1
      get_H1_norm_matrix(feop)
    else
      @unreachable
    end
  end
end

struct RBSolver{S}
  info::RBInfo
  fesolver::S
end

const RBThetaMethod = RBSolver{ThetaMethod}

get_fe_solver(s::RBSolver) = s.fesolver
get_info(s::RBSolver) = s.info

function RBSolver(fesolver,dir;kwargs...)
  info = RBInfo(dir;kwargs...)
  RBSolver(info,fesolver)
end

function fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  info = get_info(solver)
  fesolver = get_fe_solver(solver)
  nparams = num_params(info)
  sol = solve(fesolver,op,uh0;nparams)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values = collect(odesol)
  end
  snaps = Snapshots(values,realization)
  cs = ComputationalStats(stats,nparams)
  save(solver,(snaps,cs))

  return snaps,cs
end

function Algebra.solve(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  snaps,fem_stats = fe_solutions(solver,feop,uh0)
  rbop = reduced_operator(solver,feop,snaps)
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  results = rb_results(solver,rbop,snaps,rb_sol,fem_stats,rb_stats;kwargs...)
  return results
end
