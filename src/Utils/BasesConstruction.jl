get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)

function mode2_unfolding(S::AbstractMatrix,ns::Int)
  Nt = get_Nt(S,ns)
  idx_fun(ns) = (ns .- 1)*Nt .+ 1:ns*Nt
  idx = idx_fun.(1:ns)
  mode2_blocks(i) = Matrix(transpose(getindex(S,:,i)))
  mode2 = Matrix(mode2_blocks.(idx))

  mode2
end

my_svd(s::Matrix{Float}) = svd(s)

my_svd(s::SparseMatrixCSC) = svds(s;nsv=size(s)[2]-1)[1]

my_svd(s::Vector{AbstractMatrix}) = my_svd(Matrix(s))

function POD(S::NTuple{N,AbstractMatrix},args...;kwargs...) where N
  Broadcasting(si -> POD(si,args...;kwargs...))(S)
end

#= function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  Sx = S'*X*S
  Ux,Σ,_ = my_svd(Sx)
  energies = cumsum(Σ)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  U = S*Ux[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (sqrt(Σ[i])+eps())
  end
  U
end =#

function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = my_svd(L'*S[H.p,:])

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

function POD(S::AbstractMatrix;ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  U[:,1:n]
end

function POD(S::AbstractMatrix,::Val{true};ϵ=1e-5)
  reduced_POD(S;ϵ)
end

function reduced_POD(S::AbstractMatrix;ϵ=1e-5)
  reduced_POD(Val{size(S,1)>size(S,2)}(),S;ϵ)
end

function reduced_POD(::Val{true},S::AbstractMatrix;ϵ=1e-5)
  C = S'*S
  _,_,V = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  U = S*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function reduced_POD(::Val{false},S::AbstractMatrix;ϵ=1e-5)
  C = S*S'
  U,_ = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  U[:,1:n]
end

function randomized_POD(S::AbstractMatrix;ϵ=1e-5,q=1)
  randomized_POD(Val{size(S,1)>size(S,2)}(),S;ϵ,q=q)
end

function randomized_POD(::Val{true},S::AbstractMatrix;ϵ=1e-5,q=1)
  Matrix(randomized_POD(Val{false}(),S';ϵ,q=q)')
end

function randomized_POD(::Val{false},S::AbstractMatrix;ϵ=1e-5,q=1)
  Σ = svdvals(S)
  r = count(x->x>min(size(S)...)*eps()*Σ[1],Σ)
  G = rand(Normal(0.,1.),size(S,1),min(2*r,size(S,2)))
  Y = G'*(S*S')^q*S
  Q,R = qr(Y')
  B = S*Matrix(Q)
  QRb = qr(B)
  QRr = qr(R',ColumnNorm())

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)

  QRb.Q*QRr.P[:,1:n]
end

function projection(vnew::AbstractVector,v::AbstractVector;X=nothing)
  isnothing(X) ? projection(vnew,v,I(length(v))) : projection(vnew,v,X)
end

function projection(vnew::AbstractVector,v::AbstractVector,X::AbstractMatrix)
  v*(vnew'*X*v)
end

function projection(vnew::AbstractVector,basis::AbstractMatrix,X::AbstractMatrix)
  sum([projection(vnew,basis[:,i],X) for i=axes(basis,2)])
end

function orth_projection(vnew::AbstractVector,v::AbstractVector;X=nothing)
  isnothing(X) ? orth_projection(vnew,v,I(length(v))) : orth_projection(vnew,v,X)
end

function orth_projection(vnew::AbstractVector,v::AbstractVector,X::AbstractMatrix)
  projection(vnew,v,X)/(v'*X*v)
end

function orth_projection(vnew::AbstractVector,basis::AbstractMatrix,X::AbstractMatrix)
  sum([orth_projection(vnew,basis[:,i],X) for i=axes(basis,2)])
end

isbasis(basis,args...) =
  all([isapprox(norm(basis[:,j],args...),1) for j=axes(basis,2)])

function orth_complement(
  v::AbstractVector,
  basis::AbstractMatrix;
  kwargs...)

  v - orth_projection(v,basis;kwargs...)
end

function gram_schmidt(mat::Matrix{Float},basis::Matrix{Float};kwargs...)
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

function gram_schmidt(vec::Vector{Vector{T}},basis::Matrix{Float};kwargs...) where T
  gram_schmidt(Matrix(vec),basis;kwargs...)
end
