struct PTMap{F}
  f::F
  function PTMap(f::F) where F
    new{F}(f)
  end
end

function Arrays.return_value(::PTMap{F},args...) where F
  Q = map(length,args) |> first
  arg = map(testitem,args)
  intv = return_value(F(),arg...)
  fill(intv,Q)
end

function Arrays.return_cache(::PTMap{F},args...) where F
  Q = map(length,args) |> first
  _get_cache(c) = fill(c.array,Q)
  _get_cache(c::Tuple) = _get_cache(first(c))

  arg = map(testitem,args)
  intc = return_cache(F(),arg...)
  inta = _get_cache(intc)
  argc = map(array_cache,args)
  intc,inta,argc
end

function Arrays.evaluate!(cache,::PTMap{F},args...) where F
  intc,inta,argc = cache
  Q = length(inta)
  @inbounds for q = 1:Q
    argq = map((c,a) -> getindex!(c,a,q),argc,args)
    intq = evaluate!(intc,F(),argq...)
    inta[q] = copy(intq)
  end
  inta
end

# struct EvalODETrialMap <: Map
#   op::ParamODEOperator
# end

# function Arrays.return_value(k::EvalODETrialMap,args...)
#   allocate_cache(k.op)
# end

# function Arrays.return_cache(k::EvalODETrialMap,args...)
#   allocate_cache(k.op)
# end

# function Arrays.evaluate!(cache,k::EvalODETrialMap,args...)
#   update_cache!(cache,k.op,args...)
# end
