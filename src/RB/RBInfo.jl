struct RBInfo
  ϵ::Float
  nsnaps_state::Int
  nsnaps_system::Int
  nsnaps_test::Int
  fe_path::String
  rb_path::String
  energy_norm::Union{Symbol,Vector{Symbol}}
  compute_supremizers::Bool
  st_mdeim::Bool
  fun_mdeim::Bool
  load_solutions::Bool
  save_solutions::Bool
  load_structures::Bool
  save_structures::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,nsnaps_state=80,nsnaps_system=20,nsnaps_test=10,
  energy_norm=:l2,compute_supremizers=true,st_mdeim=false,fun_mdeim=false,
  load_solutions=false,save_solutions=true,load_structures=false,save_structures=true,postprocess=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim,fun_mdeim)
  RBInfo(ϵ,nsnaps_state,nsnaps_system,nsnaps_test,fe_path,rb_path,energy_norm,compute_supremizers,
    st_mdeim,fun_mdeim,load_solutions,save_solutions,load_structures,save_structures,postprocess)
end

function get_fe_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function get_rb_path(
  tpath::String,ϵ::Float;
  st_mdeim=false,fun_mdeim=false)

  @assert isdir(tpath) "Provide valid path for the current test"

  keyword = if !st_mdeim && !fun_mdeim
    "standard"
  else
    st = st_mdeim ? "st" : ""
    fun = fun_mdeim ? "fun" : ""
    st*fun
  end

  outermost_path = joinpath(tpath,"rb")
  outer_path = joinpath(outermost_path,keyword)
  rb_path = joinpath(outer_path,"$ϵ")
  create_dir!(rb_path)
  rb_path
end

for (fsave,fload) in zip((:save,:save_test),(:load,:load_test))
  @eval begin
    function $fsave(info::RBInfo,objs::Tuple)
      map(obj->$fsave(info,obj),expand(objs))
    end

    function $fload(info::RBInfo,types::Tuple)
      map(type->$fload(info,type),expand(types))
    end
end
end

function save(info::RBInfo,params::Table)
  if info.save_solutions
    path = joinpath(info.fe_path,"params")
    save(path,params)
  end
  end

function load(info::RBInfo,T::Type{Table})
  path = joinpath(info.fe_path,"params")
  load(path,T)
end

function save(info::RBInfo,stats::NamedTuple)
  if info.save_solutions
    path = joinpath(info.fe_path,"stats")
    save(path,stats)
  end
  end

function load(info::RBInfo,T::Type{NamedTuple})
  path = joinpath(info.fe_path,"stats")
  load(path,T)
end
