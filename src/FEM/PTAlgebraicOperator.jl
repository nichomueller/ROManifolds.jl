
function Base.getindex(op::NonlinearOperator,row::Int,col=:)
  feop = op.feop
  offsets = field_offsets(feop.test)
  feop_idx = feop[row,col]
  odeop_idx = get_algebraic_operator(feop_idx)
  if isa(col,Colon)
    return get_method_operator(odeop_idx,op.μ,op.t,op.dtθ,op.u0,op.ode_cache,op.vθ)
  else
    u0_idx = get_at_offsets(op.u0,offsets,col)
    vθ_idx = get_at_offsets(op.vθ,offsets,col)
    ode_cache_idx = cache_at_idx(op.ode_cache,col)
    return get_method_operator(odeop_idx,op.μ,op.t,op.dtθ,u0_idx,ode_cache_idx,vθ_idx)
  end
end

function cache_at_idx(ode_cache,idx::Int)
  _Us,_Uts,fecache = ode_cache
  Us,Uts = (),()
  for i in eachindex(_Us)
    Us = (Us...,_Us[i][idx])
    Uts = (Uts...,_Uts[i][idx])
  end
  Us,Uts,fecache
end

function update_method_operator(op::NonlinearOperator,x::AbstractVector)
  @unpack feop,μ,t,dtθ,u0,ode_cache,vθ = op
  get_method_operator(feop,μ,t,dtθ,x,ode_cache,vθ)
end
