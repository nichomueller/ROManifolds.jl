function collect_from_workers(::Type{T},sym::Symbol) where T
  local_var,Nc = _allocate_local_variable(T,sym)
  for (i,w) in enumerate(workers()[2:end])
    copyto!(view(local_var,:,i*Nc+1:(i+1)*Nc),_from_worker(sym,w))
  end
  return local_var
end

function _allocate_local_variable(::Type{T},sym::Symbol) where T
  var_from_first_worker = _from_worker(sym,first(workers()))
  Nr,Nc = size(var_from_first_worker)
  local_var = allocate_matrix(T,Nr,Nc*nworkers())
  copyto!(view(local_var,:,1:Nc),var_from_first_worker)
  return local_var,Nc
end

function _from_worker(sym::Symbol,w::Int)
  return @fetchfrom w eval(sym)
end
