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

function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = my_svd(L'*S[H.p, :])

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
  reduced_POD(S;ϵ=ϵ)
end

function reduced_POD(S::AbstractMatrix;ϵ=1e-5)
  reduced_POD(Val{size(S,1)>size(S,2)}(),S;ϵ=ϵ)
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

function iterative_reduced_POD(S::AbstractMatrix;ϵ=1e-5)
  iterative_reduced_POD(Val{size(S,1)>size(S,2)}(),S;ϵ=ϵ)
end

function iterative_reduced_POD(::Val{true},S::AbstractMatrix;ϵ=1e-5)
  C = S'*S
  _,_,V = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]

  U = S*V
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U,Σ,n
end

function iterative_reduced_POD(::Val{false},S::AbstractMatrix;ϵ=1e-5)
  C = S*S'
  U,_ = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]

  U,Σ,n
end

function randomized_POD(S::AbstractMatrix;ϵ=1e-5,q=1)
  nrow,ncol = size(S)
  r = compute_rank(S;tol=ϵ)
  gauss_mat = rand(Normal(0.,1.),ncol,2*r)
  Y = nrow>ncol ? S*(S'*S)^q*gauss_mat : (S*S')^q*S*gauss_mat
  Q,_ = qr(Y)
  B = Q'*S
  Utemp = POD(B;ϵ=ϵ)
  Q*Utemp
end

function projection(vnew::AbstractVector,v::AbstractVector)
  v*(vnew'*v)
end

function projection(vnew::AbstractVector,basis::AbstractMatrix)
  sum([projection(vnew,basis[:,i]) for i=axes(basis,2)])
end

function orth_projection(vnew::AbstractVector,v::AbstractVector)
  projection(vnew,v)/(v'*v)
end

function orth_projection(vnew::AbstractVector,basis::AbstractMatrix)
  sum([orth_projection(vnew,basis[:,i]) for i=axes(basis,2)])
end

isbasis(basis,args...) =
  all([isapprox(norm(basis[:,j],args...),1) for j=axes(basis,2)])

function orth_complement(
  v::AbstractVector,
  basis::AbstractMatrix)

  v - projection(v,basis)
end

function gram_schmidt(mat::Matrix{Float},basis::Matrix{Float})

  for i = axes(mat,2)
    mat_i = mat[:,i]
    mat_i = orth_complement(mat_i,basis)
    if i > 1
      mat_i = orth_complement(mat_i,mat[:,1:i-1])
    end
    mat[:,i] = mat_i/norm(mat_i)
  end

  mat
end

function gram_schmidt(vec::Vector{Vector},basis::Matrix{Float})
  gram_schmidt(Matrix(vec),basis)
end
