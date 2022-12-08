function assemble_rb_structure(info::RBInfo,tt::TimeTracker,op::RBVarOperator,args...)
  tt.offline_time += @elapsed begin
    rb_variable = assemble_rb_structure(info,op,args...)
  end
  save(info,rb_variable,get_id(op))
  rb_variable
end

function assemble_rb_structure(
  ::RBInfo,
  op::RBVarOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  rb_projection(op)
end

function assemble_rb_structure(
  info::RBInfo,
  op::RBVarOperator,
  μ::Vector{Param},
  args...)

  id = get_id(op)
  println("Matrix $id is non-affine: running the MDEIM offline phase on $(info.mdeim_nsnap) snapshots")

  mdeim_offline(info,op,μ,args...)
end

save(info::RBInfo,b,id::Symbol) =
  if info.save_offline save(info.offline_path,b,id) end

function save(path::String,basis::Matrix{Float},id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),basis)
end

function save(path::String,b::NTuple{2,Matrix{Float}},id::Symbol)
  basis,basis_lift = b
  save(path,basis,id)
  save(path,basis_lift,id*:_lift)
end

function save(path::String,mdeim::MDEIMSteady,id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space_"*"$id"),get_idx_space(mdeim))
  save_idx_lu_factors(path,mdeim,id)
end

function save(path::String,mdeim::MDEIMUnsteady,id::Symbol)
  save(joinpath(path,"basis_space_"*"$id"),get_basis_space(mdeim))
  save(joinpath(path,"idx_space_"*"$id"),get_idx_space(mdeim))
  save(joinpath(path,"basis_time_"*"$id"),get_basis_time(mdeim))
  save(joinpath(path,"idx_time_"*"$id"),get_idx_time(mdeim))
  save_idx_lu_factors(path,mdeim,id)
end

function save(path::String,m::NTuple{2,<:MDEIM},id::Symbol)
  mdeim,mdeim_lift = m
  save(path,mdeim,id)
  save(path,mdeim_lift,id*:_lift)
end

function load_rb_structure(
  info::RBInfo,
  op::RBVarOperator,
  meas::Measure)

  id = get_id(op)
  id_path = joinpath(info.offline_path,"basis_space_$id")
  if myisfile(id_path)
    println("Importing reduced $id")
    load_mdeim(info,op,meas)
  else
    error("Not implemented")
    #assemble_rb_structure(info,op,args...)
  end

end

function load_rb_structure(
  info::RBInfo,
  op::RBVarOperator{Affine,TT,Tsp},
  args...) where {TT,Tsp}

  id = get_id(op)
  id_path = joinpath(info.offline_path,"basis_space_$id")
  if myisfile(id_path)
    println("Importing reduced $id")
    load(id_path)
  else
    error("Not implemented")
    #assemble_rb_structure(info,op,args...)
  end

end
