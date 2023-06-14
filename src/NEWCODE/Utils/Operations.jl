function projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v) : v*sum(vnew'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v)/sum(v'*v) : v*sum(vnew'*X*v)/sum(v'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_complement(
  v::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  compl = reshape(similar(basis),:,1)
  copyto!(compl,v - orth_projection(v,basis;kwargs...))
end

function gram_schmidt(
  mat::AbstractMatrix{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  for i = axes(mat,2)
    mat_i = mat[:,i]
    mat_i = orth_complement(mat_i,basis;kwargs...)
    if i > 1
      mat_i = orth_complement(mat_i,mat[:,1:i-1];kwargs...)
    end
    mat[:,i] = mat_i/norm(mat_i)
  end

  mat
end

function expand(tup::Tuple)
  ntup = ()
  for el = tup
    if typeof(el) <: Tuple
      ntup = (ntup...,expand(el)...)
    else
      ntup = (ntup...,el)
    end
  end
  ntup
end

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(S)
  I = Vector{Ti}(undef, numnz)
  J = Vector{Ti}(undef, numnz)
  V = Vector{Tv}(undef, numnz)

  count = 1
  @inbounds for col = 1 : size(S, 2), k = SparseArrays.getcolptr(S)[col] : (SparseArrays.getcolptr(S)[col+1]-1)
      I[count] = rowvals(S)[k]
      J[count] = col
      V[count] = nonzeros(S)[k]
      count += 1
  end

  nz = findall(x -> x .>= eps(), V)

  (I[nz], J[nz], V[nz])
end

function SparseArrays.findnz(x::SparseVector{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(x)

  I = Vector{Ti}(undef, numnz)
  V = Vector{Tv}(undef, numnz)

  nzind = SparseArrays.nonzeroinds(x)
  nzval = nonzeros(x)

  @inbounds for i = 1 : numnz
      I[i] = nzind[i]
      V[i] = nzval[i]
  end

  nz = findall(v -> abs.(v) .>= eps(), V)

  (I[nz], V[nz])
end

Base.getindex(emat::EMatrix{Float},::Colon,k::Int) = emat[:,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,::Colon) = emat[k:k,:]

Base.getindex(emat::EMatrix{Float},idx::UnitRange{Int},k::Int) = emat[idx,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,idx::UnitRange{Int}) = emat[k:k,idx]

Gridap.get_triangulation(m::Measure) = m.quad.trian

# Remove when possible
function Gridap.Geometry.is_change_possible(strian::Triangulation,ttrian::Triangulation)
  if strian === ttrian
    return true
  end
  #@check get_background_model(strian) === get_background_model(ttrian)
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  is_change_possible(sglue,tglue)
end
