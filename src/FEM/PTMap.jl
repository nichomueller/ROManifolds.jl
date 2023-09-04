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
  arg = map(testitem,args)
  intc = return_cache(F(),arg...)
  inta = fill(intc.array,Q)
  argc = map(array_cache,args)
  intc,inta,argc
end

function Arrays.evaluate!(cache,::PTMap{F},args...) where F
  intc,inta,argc = cache
  Q = length(inta)
  @inbounds for q = 1:Q
    argq = map((c,a) -> getindex!(c,a,q),argc,args)
    inta[q] = evaluate!(intc,F(),argq...)
  end
  inta
end

function Arrays.return_cache(k::ConstrainRowsMap,array,constr,mask)
  return_cache(*,constr,array)
end

function Arrays.evaluate!(cache,k::ConstrainRowsMap,array,constr,mask)
  if mask
    evaluate!(cache,*,constr,array)
  else
    array
  end
end

function Arrays.return_cache(k::ConstrainRowsMap,matvec::Tuple,constr,mask)
  mat, vec = matvec
  cmat = return_cache(k,mat,constr,mask)
  cvec = return_cache(k,vec,constr,mask)
  (cmat,cvec)
end

function Arrays.evaluate!(cache,k::ConstrainRowsMap,matvec::Tuple,constr,mask)
  if mask
    cmat, cvec = cache
    mat, vec = matvec
    _mat = evaluate!(cmat,k,mat,constr,mask)
    _vec = evaluate!(cvec,k,vec,constr,mask)
    (_mat,_vec)
  else
    matvec
  end
end

function Arrays.return_cache(k::ConstrainColsMap,array,constr_t,mask)
  return_cache(*,array,constr_t)
end

function Arrays.evaluate!(cache,k::ConstrainColsMap,array,constr_t,mask)
  if mask
    evaluate!(cache,*,array,constr_t)
  else
    array
  end
end

function Arrays.return_cache(k::ConstrainColsMap,matvec::Tuple,constr_t,mask)
  mat, vec = matvec
  return_cache(k,mat,constr_t,mask)
end

function Arrays.evaluate!(cache,k::ConstrainColsMap,matvec::Tuple,constr_t,mask)
  if mask
    mat, vec = matvec
    _mat = evaluate!(cache,k,mat,constr_t,mask)
    (_mat,vec)
  else
    matvec
  end
end
