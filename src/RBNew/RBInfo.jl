function get_fe_path(tpath::String)
  fepath = joinpath(tpath,"fem")
  create_dir(fepath)
  fepath
end

function get_rb_path(tpath::String,ϵ;st_mdeim=false)
  keyword = st_mdeim ? "st" : "standard"
  outermost_path = joinpath(tpath,"rb")
  outer_path = joinpath(outermost_path,keyword)
  rb_path = joinpath(outer_path,"$ϵ")
  create_dir(rb_path)
  rb_path
end

struct RBInfo
  ϵ::Float64
  fe_path::String
  rb_path::String
  norm_style::Symbol
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  st_mdeim::Bool
end

function RBInfo(
  test_path::String;
  ϵ=1e-4,
  norm_style=:l2,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  st_mdeim=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,fe_path,rb_path,norm_style,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim)
end

num_offline_params(rbinfo::RBInfo) = rbinfo.nsnaps_state
num_online_params(rbinfo::RBInfo) = rbinfo.nsnaps_test
FEM.num_params(rbinfo::RBInfo) = num_offline_params(rbinfo) + num_online_params(rbinfo)
num_mdeim_params(rbinfo::RBInfo) = rbinfo.nsnaps_mdeim
get_tol(rbinfo::RBInfo) = rbinfo.ϵ

function get_norm_matrix(rbinfo::RBInfo,feop::TransientParamFEOperator)
  norm_style = rbinfo.norm_style
  try
    T = get_vector_type(feop.test)
    load(rbinfo,SparseMatrixCSC{eltype(T),Int};norm_style)
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

struct BlockRBInfo
  ϵ::Float64
  fe_path::String
  rb_path::String
  norm_style::Vector{Symbol}
  compute_supremizers::Bool
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  st_mdeim::Bool
end

function BlockRBInfo(
  test_path::String;
  ϵ=1e-4,
  norm_style=[:l2,:l2],
  compute_supremizers=true,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  st_mdeim=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  BlockRBInfo(ϵ,fe_path,rb_path,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim)
end

function Base.getindex(rbinfo::BlockRBInfo,i::Int)
  norm_style_i = rbinfo.norm_style[i]
  RBInfo(rbinfo.ϵ,rbinfo.fe_path,rbinfo.rb_path,norm_style_i,
    rbinfo.nsnaps_state,rbinfo.nsnaps_mdeim,rbinfo.nsnaps_test,rbinfo.st_mdeim)
end

struct ComputationInfo
  avg_time::Float64
  avg_nallocs::Float64
  function ComputationInfo(stats::NamedTuple,nruns::Int)
    avg_time = stats[:time] / nruns
    avg_nallocs = stats[:bytes] / (1e6*nruns)
    new(avg_time,avg_nallocs)
  end
end

get_avg_time(cinfo::ComputationInfo) = cinfo.avg_time
get_avg_nallocs(cinfo::ComputationInfo) = cinfo.avg_nallocs
